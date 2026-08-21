use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use aarambh_studio_core::{AarambhError, Result};
use candle_core::Tensor;
use image::{RgbImage, imageops::FilterType};
use mp4::{MediaType, Mp4Reader, TrackType};
use openh264::decoder::Decoder;
use openh264::formats::YUVSource;
use serde::{Deserialize, Serialize};

/// Strategy used to choose a fixed number of frames from a video.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum FrameSamplingStrategy {
    /// Select frames at evenly spaced positions, including both endpoints.
    #[default]
    Uniform,
    /// Prefer high visual-difference boundaries, then fill remaining positions uniformly.
    SceneAware,
}

/// Native video decode and frame-selection configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(default)]
pub struct VideoSamplingConfig {
    /// Number of frames returned for each video.
    pub frame_count: usize,
    /// Hard upper bound accepted from command-line or data configuration.
    pub max_frame_count: usize,
    /// Frame selection strategy.
    pub strategy: FrameSamplingStrategy,
    /// Minimum source-frame distance between scene-aware selections.
    pub scene_min_gap: usize,
}

impl Default for VideoSamplingConfig {
    fn default() -> Self {
        Self {
            frame_count: 8,
            max_frame_count: 8,
            strategy: FrameSamplingStrategy::Uniform,
            scene_min_gap: 8,
        }
    }
}

impl VideoSamplingConfig {
    /// Validate frame sampling limits.
    pub fn validate(&self) -> Result<()> {
        if self.frame_count == 0 || self.max_frame_count == 0 {
            return Err(AarambhError::Config(
                "video frame_count and max_frame_count must be non-zero".into(),
            ));
        }
        if self.frame_count > self.max_frame_count {
            return Err(AarambhError::Config(format!(
                "video frame_count {} exceeds max_frame_count {}",
                self.frame_count, self.max_frame_count
            )));
        }
        if self.strategy == FrameSamplingStrategy::SceneAware && self.scene_min_gap == 0 {
            return Err(AarambhError::Config(
                "video scene_min_gap must be non-zero for scene-aware sampling".into(),
            ));
        }
        Ok(())
    }
}

/// Decoded and chronologically ordered video frames.
#[derive(Debug, Clone)]
pub struct SampledVideo {
    /// RGB frames selected from the source video.
    pub frames: Vec<RgbImage>,
    /// Zero-based decoded-frame indices corresponding to `frames`.
    pub indices: Vec<usize>,
    /// Total number of decoded source frames.
    pub source_frame_count: usize,
}

/// Decode H.264 in an MP4 container and return a deterministic frame sample.
///
/// The decoder is bundled through OpenH264; no system FFmpeg installation is used.
pub fn decode_sampled_video(
    path: impl AsRef<Path>,
    config: &VideoSamplingConfig,
) -> Result<SampledVideo> {
    config.validate()?;
    let path = path.as_ref();
    if !path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("mp4"))
    {
        return Err(AarambhError::Unsupported(format!(
            "native video decoding currently supports H.264 MP4 input, got {}",
            path.display()
        )));
    }

    match config.strategy {
        FrameSamplingStrategy::Uniform => decode_uniform(path, config.frame_count),
        FrameSamplingStrategy::SceneAware => decode_scene_aware(path, config),
    }
}

fn decode_uniform(path: &Path, target: usize) -> Result<SampledVideo> {
    let (mut reader, track) = open_h264(path)?;
    let expected = reader.sample_count(track.track_id).map_err(video_error)? as usize;
    let wanted = uniform_frame_indices(expected.max(1), target);
    let mut frames = Vec::with_capacity(target);
    let mut indices = Vec::with_capacity(target);
    let mut last = None;
    let mut next_wanted = 0usize;
    decode_frames(&mut reader, &track, |index, frame| {
        while next_wanted < wanted.len() && wanted[next_wanted] == index {
            frames.push(frame.clone());
            indices.push(index);
            next_wanted += 1;
        }
        last = Some((index, frame));
        Ok(())
    })?;
    let source_frame_count = last.as_ref().map_or(0, |(index, _)| index + 1);
    if source_frame_count == 0 {
        return Err(AarambhError::Config(format!(
            "video {} decoded no frames",
            path.display()
        )));
    }
    fill_sample(&mut frames, &mut indices, target, last.as_ref());
    Ok(SampledVideo {
        frames,
        indices,
        source_frame_count,
    })
}

fn decode_scene_aware(path: &Path, config: &VideoSamplingConfig) -> Result<SampledVideo> {
    let (mut first_reader, first_track) = open_h264(path)?;
    let mut thumbnails = Vec::new();
    decode_frames(&mut first_reader, &first_track, |_index, frame| {
        thumbnails.push(luma_thumbnail(&frame));
        Ok(())
    })?;
    if thumbnails.is_empty() {
        return Err(AarambhError::Config(format!(
            "video {} decoded no frames",
            path.display()
        )));
    }
    let wanted = scene_aware_frame_indices(&thumbnails, config.frame_count, config.scene_min_gap);
    let (mut reader, track) = open_h264(path)?;
    let mut frames = Vec::with_capacity(config.frame_count);
    let mut indices = Vec::with_capacity(config.frame_count);
    let mut last = None;
    decode_frames(&mut reader, &track, |index, frame| {
        if wanted.binary_search(&index).is_ok() {
            frames.push(frame.clone());
            indices.push(index);
        }
        last = Some((index, frame));
        Ok(())
    })?;
    fill_sample(&mut frames, &mut indices, config.frame_count, last.as_ref());
    Ok(SampledVideo {
        frames,
        indices,
        source_frame_count: thumbnails.len(),
    })
}

struct H264Track {
    track_id: u32,
    sample_count: u32,
    length_size: usize,
    parameter_sets: Vec<Vec<u8>>,
}

fn open_h264(path: &Path) -> Result<(Mp4Reader<BufReader<File>>, H264Track)> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let reader = BufReader::new(file);
    let mp4 = Mp4Reader::read_header(reader, size).map_err(video_error)?;
    let (track_id, track) = mp4
        .tracks()
        .iter()
        .find(|(_, track)| track.track_type().ok() == Some(TrackType::Video))
        .ok_or_else(|| AarambhError::Config("MP4 contains no video track".into()))?;
    let media_type = track.media_type().map_err(video_error)?;
    if media_type != MediaType::H264 {
        return Err(AarambhError::Unsupported(format!(
            "native decoder supports H.264 MP4 tracks, found {media_type}"
        )));
    }
    let avc = track
        .trak
        .mdia
        .minf
        .stbl
        .stsd
        .avc1
        .as_ref()
        .ok_or_else(|| AarambhError::Config("H.264 track is missing avcC metadata".into()))?;
    let parameter_sets = avc
        .avcc
        .sequence_parameter_sets
        .iter()
        .chain(avc.avcc.picture_parameter_sets.iter())
        .map(|nal| nal.bytes.clone())
        .collect();
    let metadata = H264Track {
        track_id: *track_id,
        sample_count: track.sample_count(),
        length_size: avc.avcc.length_size_minus_one as usize + 1,
        parameter_sets,
    };
    Ok((mp4, metadata))
}

fn decode_frames(
    reader: &mut Mp4Reader<BufReader<File>>,
    track: &H264Track,
    mut receive: impl FnMut(usize, RgbImage) -> Result<()>,
) -> Result<()> {
    let mut decoder = Decoder::new().map_err(video_error)?;
    for nal in &track.parameter_sets {
        let _ = decoder.decode(&annex_b_nal(nal)).map_err(video_error)?;
    }
    let mut decoded_index = 0usize;
    for sample_id in 1..=track.sample_count {
        let Some(sample) = reader
            .read_sample(track.track_id, sample_id)
            .map_err(video_error)?
        else {
            continue;
        };
        for nal in split_length_prefixed(sample.bytes.as_ref(), track.length_size)? {
            let Some(yuv) = decoder.decode(&annex_b_nal(nal)).map_err(video_error)? else {
                continue;
            };
            let (width, height) = yuv.dimensions();
            let mut rgb = vec![0u8; yuv.rgb8_len()];
            yuv.write_rgb8(&mut rgb);
            let frame = RgbImage::from_raw(width as u32, height as u32, rgb).ok_or_else(|| {
                AarambhError::Shape("decoded RGB frame has an invalid byte length".into())
            })?;
            receive(decoded_index, frame)?;
            decoded_index += 1;
        }
    }
    Ok(())
}

fn split_length_prefixed(data: &[u8], length_size: usize) -> Result<Vec<&[u8]>> {
    if !(1..=4).contains(&length_size) {
        return Err(AarambhError::Config(format!(
            "invalid H.264 NAL length size {length_size}"
        )));
    }
    let mut nals = Vec::new();
    let mut offset = 0usize;
    while offset < data.len() {
        if offset + length_size > data.len() {
            return Err(AarambhError::Config(
                "truncated H.264 NAL length prefix".into(),
            ));
        }
        let mut length = 0usize;
        for byte in &data[offset..offset + length_size] {
            length = (length << 8) | *byte as usize;
        }
        offset += length_size;
        let end = offset
            .checked_add(length)
            .ok_or_else(|| AarambhError::Config("H.264 NAL length overflow".into()))?;
        if end > data.len() {
            return Err(AarambhError::Config("truncated H.264 NAL payload".into()));
        }
        if length != 0 {
            nals.push(&data[offset..end]);
        }
        offset = end;
    }
    Ok(nals)
}

fn annex_b_nal(nal: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(nal.len() + 4);
    packet.extend_from_slice(&[0, 0, 0, 1]);
    packet.extend_from_slice(nal);
    packet
}

/// Return endpoint-inclusive, monotonically ordered uniform frame indices.
pub fn uniform_frame_indices(source_count: usize, target_count: usize) -> Vec<usize> {
    assert!(source_count > 0, "source_count must be non-zero");
    assert!(target_count > 0, "target_count must be non-zero");
    if target_count == 1 {
        return vec![(source_count - 1) / 2];
    }
    (0..target_count)
        .map(|index| {
            let numerator = index * (source_count - 1);
            (numerator + (target_count - 1) / 2) / (target_count - 1)
        })
        .collect()
}

/// Select scene boundaries from flattened luma thumbnails.
pub fn scene_aware_frame_indices(
    thumbnails: &[Vec<u8>],
    target_count: usize,
    min_gap: usize,
) -> Vec<usize> {
    assert!(!thumbnails.is_empty(), "thumbnails must be non-empty");
    assert!(target_count > 0, "target_count must be non-zero");
    if thumbnails.len() == 1 {
        return vec![0; target_count];
    }
    let mut selected = vec![0, thumbnails.len() - 1];
    let mut changes = (1..thumbnails.len())
        .map(|index| {
            (
                mean_absolute_difference(&thumbnails[index - 1], &thumbnails[index]),
                index,
            )
        })
        .collect::<Vec<_>>();
    changes.sort_by(|left, right| right.0.total_cmp(&left.0).then(left.1.cmp(&right.1)));
    for (_, index) in changes {
        if selected.len() >= target_count {
            break;
        }
        if selected
            .iter()
            .all(|chosen| chosen.abs_diff(index) >= min_gap)
        {
            selected.push(index);
        }
    }
    for index in uniform_frame_indices(thumbnails.len(), target_count) {
        if selected.len() >= target_count {
            break;
        }
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    selected.sort_unstable();
    while selected.len() < target_count {
        selected.push(*selected.last().unwrap_or(&0));
    }
    selected.truncate(target_count);
    selected
}

fn luma_thumbnail(frame: &RgbImage) -> Vec<u8> {
    let thumbnail = image::imageops::resize(frame, 32, 32, FilterType::Triangle);
    thumbnail
        .pixels()
        .map(|pixel| {
            ((77u16 * pixel[0] as u16 + 150u16 * pixel[1] as u16 + 29u16 * pixel[2] as u16) >> 8)
                as u8
        })
        .collect()
}

fn mean_absolute_difference(left: &[u8], right: &[u8]) -> f32 {
    if left.len() != right.len() || left.is_empty() {
        return 0.0;
    }
    let sum = left
        .iter()
        .zip(right)
        .map(|(a, b)| a.abs_diff(*b) as u64)
        .sum::<u64>();
    sum as f32 / left.len() as f32
}

fn fill_sample(
    frames: &mut Vec<RgbImage>,
    indices: &mut Vec<usize>,
    target: usize,
    last: Option<&(usize, RgbImage)>,
) {
    if frames.is_empty()
        && let Some((index, frame)) = last
    {
        frames.push(frame.clone());
        indices.push(*index);
    }
    while frames.len() < target {
        frames.push(frames.last().expect("at least one decoded frame").clone());
        indices.push(*indices.last().expect("at least one decoded frame index"));
    }
    frames.truncate(target);
    indices.truncate(target);
}

fn video_error(error: impl std::fmt::Display) -> AarambhError {
    AarambhError::Config(format!("video decode failed: {error}"))
}

/// Key used by the bounded frozen-encoder feature cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VideoFeatureCacheKey {
    path: PathBuf,
    modified_nanos: u128,
    file_size: u64,
    sampling: VideoSamplingConfig,
    encoder_signature: String,
}

impl VideoFeatureCacheKey {
    /// Build a cache key from file metadata and preprocessing configuration.
    pub fn new(
        path: impl AsRef<Path>,
        sampling: VideoSamplingConfig,
        encoder_signature: impl Into<String>,
    ) -> Result<Self> {
        let path = path.as_ref().canonicalize()?;
        let metadata = path.metadata()?;
        let modified_nanos = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_nanos());
        Ok(Self {
            path,
            modified_nanos,
            file_size: metadata.len(),
            sampling,
            encoder_signature: encoder_signature.into(),
        })
    }
}

/// Bounded FIFO cache for detached, pre-projector CLIP frame features.
#[derive(Debug)]
pub struct VideoFeatureCache {
    capacity: usize,
    entries: HashMap<VideoFeatureCacheKey, Tensor>,
    order: VecDeque<VideoFeatureCacheKey>,
}

impl VideoFeatureCache {
    /// Create a cache with the given maximum number of videos.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    /// Return cached frozen-encoder features when present.
    pub fn get(&self, key: &VideoFeatureCacheKey) -> Option<Tensor> {
        self.entries.get(key).cloned()
    }

    /// Insert detached frozen features and evict the oldest key when full.
    pub fn insert(&mut self, key: VideoFeatureCacheKey, features: Tensor) {
        if self.capacity == 0 {
            return;
        }
        if let std::collections::hash_map::Entry::Occupied(mut entry) =
            self.entries.entry(key.clone())
        {
            entry.insert(features);
            return;
        }
        while self.entries.len() >= self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(key, features);
    }

    /// Return the number of cached videos.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether no videos are cached.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_sampling_is_endpoint_inclusive_and_exact() {
        assert_eq!(uniform_frame_indices(10, 4), vec![0, 3, 6, 9]);
        assert_eq!(uniform_frame_indices(2, 4), vec![0, 0, 1, 1]);
        assert_eq!(uniform_frame_indices(9, 1), vec![4]);
    }

    #[test]
    fn scene_sampling_prioritizes_largest_change() {
        let mut frames = vec![vec![0; 16]; 6];
        frames[3].fill(255);
        let selected = scene_aware_frame_indices(&frames, 4, 1);
        assert_eq!(selected.len(), 4);
        assert_eq!(selected[0], 0);
        assert_eq!(*selected.last().unwrap(), 5);
        assert!(selected.contains(&3));
    }

    #[test]
    fn length_prefixed_nals_are_validated() {
        let data = [0, 0, 0, 2, 0x67, 1, 0, 0, 0, 1, 0x68];
        let nals = split_length_prefixed(&data, 4).unwrap();
        assert_eq!(nals, vec![&data[4..6], &data[10..11]]);
        assert!(split_length_prefixed(&data[..5], 4).is_err());
    }
}
