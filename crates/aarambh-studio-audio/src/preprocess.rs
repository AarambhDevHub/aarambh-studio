//! Pure-Rust audio decode and mel-spectrogram extraction.
//!
//! Phase 42 follows the same dependency discipline v3 §35 established for video
//! container decode: local, pure-Rust decode only — no Python audio ML tooling,
//! no system-library FFI. WAV PCM (8/16/24/32-bit signed and unsigned, plus
//! 32-bit float) is decoded in-process, resampled to the configured sample rate,
//! and converted into a log-mel spectrogram with a from-scratch Hann window,
//! radix-2 Cooley-Tukey FFT, and triangular mel filterbank.

use std::path::Path;

use aarambh_studio_core::{AarambhError, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration for mel-spectrogram extraction.
///
/// Produces a `[n_mels, time_frames]` log-power mel spectrogram, the input the
/// frozen audio encoder consumes (treated as a single-channel image whose rows
/// are mel frequency bins and whose columns are short-time frames).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct MelSpectrogramConfig {
    /// Target sample rate in hertz; input audio is resampled to this rate.
    pub sample_rate: u32,
    /// FFT window length in samples; must be a power of two.
    pub n_fft: usize,
    /// Hop length between successive short-time windows in samples.
    pub hop_length: usize,
    /// Number of triangular mel filterbank bins.
    pub n_mels: usize,
    /// Lowest filterbank frequency in hertz.
    pub fmin: f32,
    /// Highest filterbank frequency in hertz.
    pub fmax: f32,
    /// Maximum source audio duration in seconds; longer clips are truncated.
    pub max_audio_seconds: f32,
    /// Spectrogram power exponent (`2.0` for power, `1.0` for magnitude).
    pub power: f32,
    /// Log offset added before the natural log to avoid `log(0)`.
    pub log_eps: f32,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8_000.0,
            max_audio_seconds: 10.0,
            power: 2.0,
            log_eps: 1.0e-10,
        }
    }
}

impl MelSpectrogramConfig {
    /// Validate mel-spectrogram parameters.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0
            || self.n_fft == 0
            || self.hop_length == 0
            || self.n_mels == 0
            || self.max_audio_seconds <= 0.0
        {
            return Err(AarambhError::Config(
                "mel sample_rate, n_fft, hop_length, n_mels, and max_audio_seconds must be non-zero"
                    .into(),
            ));
        }
        if !self.n_fft.is_power_of_two() {
            return Err(AarambhError::Config(format!(
                "mel n_fft {} must be a power of two",
                self.n_fft
            )));
        }
        if self.hop_length > self.n_fft {
            return Err(AarambhError::Config(format!(
                "mel hop_length {} must not exceed n_fft {}",
                self.hop_length, self.n_fft
            )));
        }
        if self.fmax <= self.fmin {
            return Err(AarambhError::Config(format!(
                "mel fmax {} must exceed fmin {}",
                self.fmax, self.fmin
            )));
        }
        if self.power <= 0.0 || self.log_eps <= 0.0 {
            return Err(AarambhError::Config(
                "mel power and log_eps must be positive".into(),
            ));
        }
        Ok(())
    }

    /// Number of short-time frames produced for the maximum audio duration.
    pub fn max_frames(&self) -> usize {
        let total_samples = (self.max_audio_seconds * self.sample_rate as f32) as usize;
        if total_samples < self.n_fft {
            return 1;
        }
        (total_samples - self.n_fft) / self.hop_length + 1
    }
}

/// Audio preprocessor that decodes WAV files and extracts log-mel spectrograms.
#[derive(Debug, Clone)]
pub struct AudioPreprocessor {
    config: MelSpectrogramConfig,
    window: Vec<f32>,
    mel: Vec<Vec<f32>>,
}

impl AudioPreprocessor {
    /// Create a preprocessor from explicit configuration.
    pub fn new(config: MelSpectrogramConfig) -> Result<Self> {
        config.validate()?;
        let window = hann_window(config.n_fft);
        let mel = mel_filterbank(
            config.n_mels,
            config.n_fft,
            config.sample_rate,
            config.fmin,
            config.fmax,
        );
        Ok(Self {
            config,
            window,
            mel,
        })
    }

    /// Return the preprocessing configuration.
    pub fn config(&self) -> &MelSpectrogramConfig {
        &self.config
    }

    /// Decode a WAV file and return a normalized `[n_mels, time_frames]` tensor.
    pub fn preprocess_path(&self, path: impl AsRef<Path>, device: &Device) -> Result<Tensor> {
        let decoded = decode_wav_mono(path.as_ref())?;
        let resampled = resample_to_target(&decoded.samples, decoded.sample_rate, &self.config);
        self.mel_tensor_from_samples(&resampled, device)
    }

    /// Decode an in-memory WAV byte buffer and return a normalized mel tensor.
    pub fn preprocess_bytes(&self, bytes: &[u8], device: &Device) -> Result<Tensor> {
        let decoded = decode_wav_mono_bytes(bytes)?;
        let resampled = resample_to_target(&decoded.samples, decoded.sample_rate, &self.config);
        self.mel_tensor_from_samples(&resampled, device)
    }

    fn mel_tensor_from_samples(&self, samples: &[f32], device: &Device) -> Result<Tensor> {
        let spectrogram = self.mel_spectrogram(samples);
        let n_mels = self.config.n_mels;
        let max_frames = self.config.max_frames();
        let frames = spectrogram.len() / n_mels;
        let mut values = vec![0f32; n_mels * max_frames];
        for mel_idx in 0..n_mels {
            for frame_idx in 0..frames.min(max_frames) {
                values[mel_idx * max_frames + frame_idx] =
                    spectrogram[frame_idx * n_mels + mel_idx];
            }
        }
        Ok(Tensor::from_vec(values, (n_mels, max_frames), device)?)
    }

    fn mel_spectrogram(&self, samples: &[f32]) -> Vec<f32> {
        let MelSpectrogramConfig {
            n_fft,
            hop_length,
            n_mels,
            power,
            log_eps,
            ..
        } = self.config;
        if samples.len() < n_fft {
            let mut padded = samples.to_vec();
            padded.resize(n_fft, 0.0);
            return self.frame_to_mel(&padded, n_mels, power, log_eps);
        }
        let max_frames = self.config.max_frames();
        let mut out = Vec::with_capacity(max_frames * n_mels);
        let mut frame = vec![0f32; n_fft];
        let mut produced = 0usize;
        let mut start = 0usize;
        while start + n_fft <= samples.len() && produced < max_frames {
            for (idx, value) in frame.iter_mut().enumerate() {
                *value = samples[start + idx] * self.window[idx];
            }
            let mel = self.frame_to_mel(&frame, n_mels, power, log_eps);
            out.extend_from_slice(&mel);
            produced += 1;
            start += hop_length;
        }
        if produced == 0 {
            frame.copy_from_slice(&samples[..n_fft]);
            for (idx, value) in frame.iter_mut().enumerate() {
                *value *= self.window[idx];
            }
            out.extend_from_slice(&self.frame_to_mel(&frame, n_mels, power, log_eps));
        }
        out
    }

    fn frame_to_mel(&self, frame: &[f32], n_mels: usize, power: f32, log_eps: f32) -> Vec<f32> {
        let mut spectrum = fft_power(frame);
        let mut mel = vec![0f32; n_mels];
        for (mel_idx, filter) in self.mel.iter().enumerate().take(n_mels) {
            let mut acc = 0.0f32;
            for (bin_idx, &weight) in filter.iter().enumerate() {
                if weight > 0.0 {
                    acc += weight * spectrum[bin_idx];
                }
            }
            mel[mel_idx] = (acc.powf(power) + log_eps).ln();
        }
        // local mean / std normalization so the frozen encoder sees a stable scale
        let mean = mel.iter().sum::<f32>() / mel.len() as f32;
        let var = mel.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / mel.len() as f32;
        let std = var.sqrt().max(1.0e-6);
        for value in &mut mel {
            *value = (*value - mean) / std;
        }
        spectrum.clear();
        mel
    }
}

impl Default for AudioPreprocessor {
    fn default() -> Self {
        Self::new(MelSpectrogramConfig::default()).expect("default mel config")
    }
}

fn hann_window(n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|idx| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * idx as f32 / (n as f32 - 1.0)).cos())
        .collect()
}

fn fft_power(frame: &[f32]) -> Vec<f32> {
    let n = frame.len();
    let mut real = frame.to_vec();
    let mut imag = vec![0.0f32; n];
    fft_in_place(&mut real, &mut imag);
    let half = n / 2;
    (0..=half)
        .map(|idx| {
            let re = real[idx];
            let im = imag[idx];
            re * re + im * im
        })
        .collect()
}

/// Iterative radix-2 Cooley-Tukey FFT operating in place on real/imag pairs.
fn fft_in_place(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    if n <= 1 {
        return;
    }
    bit_reverse_permute(real, imag);
    let mut length = 2usize;
    while length <= n {
        let theta = -2.0 * std::f32::consts::PI / length as f32;
        let w_re = theta.cos();
        let w_im = theta.sin();
        let half = length / 2;
        let mut group_start = 0usize;
        while group_start < n {
            let mut cur_re = 1.0f32;
            let mut cur_im = 0.0f32;
            for pair in 0..half {
                let even = group_start + pair;
                let odd = even + half;
                let t_re = cur_re * real[odd] - cur_im * imag[odd];
                let t_im = cur_re * imag[odd] + cur_im * real[odd];
                real[odd] = real[even] - t_re;
                imag[odd] = imag[even] - t_im;
                real[even] += t_re;
                imag[even] += t_im;
                let next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
                if pair + 1 == half {
                    break;
                }
            }
            group_start += length;
        }
        length *= 2;
    }
}

fn bit_reverse_permute(real: &mut [f32], imag: &mut [f32]) {
    let n = real.len();
    let bits = n.trailing_zeros() as usize;
    for idx in 1..n {
        let mut rev = 0usize;
        let mut value = idx;
        for _ in 0..bits {
            rev = (rev << 1) | (value & 1);
            value >>= 1;
        }
        if rev > idx {
            real.swap(idx, rev);
            imag.swap(idx, rev);
        }
    }
}

fn mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sample_rate: u32,
    fmin: f32,
    fmax: f32,
) -> Vec<Vec<f32>> {
    let half = n_fft / 2 + 1;
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|idx| mel_min + (mel_max - mel_min) * idx as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|m| mel_to_hz(*m)).collect();
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|hz| (hz * n_fft as f32 / sample_rate as f32).round())
        .collect();
    let mut filters = vec![vec![0.0f32; half]; n_mels];
    for mel_idx in 0..n_mels {
        let left = bin_points[mel_idx];
        let center = bin_points[mel_idx + 1];
        let right = bin_points[mel_idx + 2];
        let end = (right as usize).min(half.saturating_sub(1));
        for (bin, slot) in filters[mel_idx]
            .iter_mut()
            .enumerate()
            .take(end + 1)
            .skip(left as usize)
        {
            let bin_f = bin as f32;
            let rising = if center > left {
                (bin_f - left) / (center - left)
            } else {
                0.0
            };
            let falling = if right > center {
                (right - bin_f) / (right - center)
            } else {
                0.0
            };
            let weight = rising.min(falling).max(0.0);
            *slot = weight;
        }
    }
    filters
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32).powf(mel / 2595.0) - 700.0
}

fn resample_to_target(
    samples: &[f32],
    source_rate: u32,
    config: &MelSpectrogramConfig,
) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let max_samples = (config.max_audio_seconds * config.sample_rate as f32) as usize;
    let resampled = if source_rate == config.sample_rate {
        samples.to_vec()
    } else {
        let ratio = config.sample_rate as f64 / source_rate as f64;
        let output_len = ((samples.len() as f64) * ratio).round() as usize;
        (0..output_len)
            .map(|idx| {
                let source_pos = idx as f64 / ratio;
                let left = source_pos.floor() as usize;
                let right = (left + 1).min(samples.len() - 1);
                let frac = source_pos - left as f64;
                samples[left] as f64 * (1.0 - frac) + samples[right] as f64 * frac
            })
            .map(|value| value as f32)
            .collect::<Vec<_>>()
    };
    if resampled.len() > max_samples {
        resampled[..max_samples].to_vec()
    } else {
        resampled
    }
}

/// Decoded mono PCM samples and their source sample rate.
#[derive(Debug, Clone)]
struct DecodedAudio {
    samples: Vec<f32>,
    sample_rate: u32,
}

fn decode_wav_mono(path: &Path) -> Result<DecodedAudio> {
    let bytes = std::fs::read(path).map_err(|err| {
        AarambhError::Io(std::io::Error::new(
            err.kind(),
            format!("failed to read audio {}: {err}", path.display()),
        ))
    })?;
    decode_wav_mono_bytes(&bytes)
}

fn decode_wav_mono_bytes(bytes: &[u8]) -> Result<DecodedAudio> {
    let mut cursor = ByteCursor::new(bytes);
    let riff = cursor.read_tag()?;
    if riff != *b"RIFF" {
        return Err(AarambhError::Config(format!(
            "audio is not a RIFF/WAVE container; got tag {:?}",
            riff
        )));
    }
    let _riff_size = cursor.read_u32_le()?;
    let wave = cursor.read_tag()?;
    if wave != *b"WAVE" {
        return Err(AarambhError::Config(format!(
            "audio RIFF container is not WAVE; got {:?}",
            wave
        )));
    }
    let mut fmt: Option<FmtChunk> = None;
    let mut data: Option<Vec<u8>> = None;
    while cursor.remaining() >= 8 {
        let tag = cursor.read_tag()?;
        let size = cursor.read_u32_le()? as usize;
        match &tag {
            b"fmt " => {
                fmt = Some(parse_fmt_chunk(cursor.read_bytes(size)?)?);
            }
            b"data" => {
                data = Some(cursor.read_bytes(size)?.to_vec());
            }
            _ => {
                cursor.skip(size)?;
            }
        }
        if size % 2 == 1 {
            cursor.skip(1)?;
        }
    }
    let fmt = fmt.ok_or_else(|| AarambhError::Config("WAV missing fmt chunk".into()))?;
    let data = data.ok_or_else(|| AarambhError::Config("WAV missing data chunk".into()))?;
    let samples = decode_pcm(&data, &fmt)?;
    Ok(DecodedAudio {
        samples,
        sample_rate: fmt.sample_rate,
    })
}

#[derive(Debug, Clone, Copy)]
struct FmtChunk {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}

fn parse_fmt_chunk(bytes: &[u8]) -> Result<FmtChunk> {
    let mut cursor = ByteCursor::new(bytes);
    let audio_format = cursor.read_u16_le()?;
    let channels = cursor.read_u16_le()?;
    let sample_rate = cursor.read_u32_le()?;
    let _byte_rate = cursor.read_u32_le()?;
    let _block_align = cursor.read_u16_le()?;
    let bits_per_sample = cursor.read_u16_le()?;
    Ok(FmtChunk {
        audio_format,
        channels,
        sample_rate,
        bits_per_sample,
    })
}

fn decode_pcm(data: &[u8], fmt: &FmtChunk) -> Result<Vec<f32>> {
    if fmt.channels == 0 {
        return Err(AarambhError::Config("WAV has zero channels".into()));
    }
    let bytes_per_sample = (fmt.bits_per_sample as usize).div_ceil(8);
    let frame_bytes = bytes_per_sample * fmt.channels as usize;
    if frame_bytes == 0 {
        return Err(AarambhError::Config(
            "WAV bits_per_sample and channels imply zero frame bytes".into(),
        ));
    }
    let frame_count = data.len() / frame_bytes;
    let mut mono = Vec::with_capacity(frame_count);
    for frame in 0..frame_count {
        let base = frame * frame_bytes;
        let mut sum = 0.0f32;
        for channel in 0..fmt.channels as usize {
            let offset = base + channel * bytes_per_sample;
            sum += decode_sample(&data[offset..offset + bytes_per_sample], fmt)?;
        }
        mono.push(sum / fmt.channels as f32);
    }
    Ok(mono)
}

fn decode_sample(bytes: &[u8], fmt: &FmtChunk) -> Result<f32> {
    match (fmt.audio_format, fmt.bits_per_sample) {
        (1, 8) => Ok((bytes[0] as i32 - 128) as f32 / 128.0),
        (1, 16) => Ok(read_i16_le(bytes) as f32 / 32_768.0),
        (1, 24) => Ok(read_i24_le(bytes) as f32 / 8_388_608.0),
        (1, 32) => Ok(read_i32_le(bytes) as f32 / 2_147_483_648.0),
        (3, 32) => Ok(read_f32_le(bytes)),
        (3, 64) => Ok(read_f64_le(bytes) as f32),
        _ => Err(AarambhError::Config(format!(
            "unsupported WAV format {}/{}-bit",
            fmt.audio_format, fmt.bits_per_sample
        ))),
    }
}

fn read_i16_le(bytes: &[u8]) -> i16 {
    i16::from_le_bytes([bytes[0], bytes[1]])
}

fn read_i24_le(bytes: &[u8]) -> i32 {
    let unsigned = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
    let value = unsigned as i32;
    if value & 0x0080_0000 != 0 {
        value | 0xFF00_0000u32 as i32
    } else {
        value
    }
}

fn read_i32_le(bytes: &[u8]) -> i32 {
    i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

fn read_f64_le(bytes: &[u8]) -> f64 {
    f64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

struct ByteCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> ByteCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn read_tag(&mut self) -> Result<[u8; 4]> {
        if self.remaining() < 4 {
            return Err(AarambhError::Config("truncated WAV tag".into()));
        }
        let tag = [
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
        ];
        self.pos += 4;
        Ok(tag)
    }

    fn read_u32_le(&mut self) -> Result<u32> {
        if self.remaining() < 4 {
            return Err(AarambhError::Config("truncated WAV u32".into()));
        }
        let value = u32::from_le_bytes([
            self.bytes[self.pos],
            self.bytes[self.pos + 1],
            self.bytes[self.pos + 2],
            self.bytes[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(value)
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        if self.remaining() < 2 {
            return Err(AarambhError::Config("truncated WAV u16".into()));
        }
        let value = u16::from_le_bytes([self.bytes[self.pos], self.bytes[self.pos + 1]]);
        self.pos += 2;
        Ok(value)
    }

    fn read_bytes(&mut self, count: usize) -> Result<&'a [u8]> {
        if self.remaining() < count {
            return Err(AarambhError::Config(format!(
                "truncated WAV chunk: requested {count} bytes, have {}",
                self.remaining()
            )));
        }
        let slice = &self.bytes[self.pos..self.pos + count];
        self.pos += count;
        Ok(slice)
    }

    fn skip(&mut self, count: usize) -> Result<()> {
        if self.remaining() < count {
            return Err(AarambhError::Config("truncated WAV padding".into()));
        }
        self.pos += count;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_wav_bytes(sample_rate: u32, frequency: f32, duration_secs: f32) -> Vec<u8> {
        let sample_count = (duration_secs * sample_rate as f32) as usize;
        let mut data = Vec::with_capacity(sample_count * 2);
        for idx in 0..sample_count {
            let t = idx as f32 / sample_rate as f32;
            let value = (2.0 * std::f32::consts::PI * frequency * t).sin();
            let pcm = (value * 32_767.0) as i16;
            data.extend_from_slice(&pcm.to_le_bytes());
        }
        let mut bytes = Vec::new();
        let riff_size = 12 + 24 + 8 + data.len() as u32;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
        bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 2;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&data);
        bytes
    }

    #[test]
    fn mel_config_max_frames_is_consistent() {
        let config = MelSpectrogramConfig {
            sample_rate: 16_000,
            n_fft: 512,
            hop_length: 256,
            n_mels: 80,
            fmin: 0.0,
            fmax: 8000.0,
            max_audio_seconds: 1.0,
            power: 2.0,
            log_eps: 1.0e-10,
        };
        config.validate().unwrap();
        // 16000 samples, 512 window, 256 hop -> (16000-512)/256 + 1 = 61 frames
        assert_eq!(config.max_frames(), 61);
    }

    #[test]
    fn fft_recovers_known_frequency() {
        let n = 512usize;
        let frequency = 4usize;
        let frame: Vec<f32> = (0..n)
            .map(|idx| {
                (2.0 * std::f32::consts::PI * frequency as f32 * idx as f32 / n as f32).sin()
            })
            .collect();
        let spectrum = fft_power(&frame);
        let peak = spectrum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(peak, frequency);
    }

    #[test]
    fn preprocess_path_returns_expected_mel_shape() {
        let bytes = synth_wav_bytes(16_000, 440.0, 0.5);
        let temp = std::env::temp_dir().join("aarambh_audio_smoke.wav");
        std::fs::write(&temp, &bytes).unwrap();
        let config = MelSpectrogramConfig {
            sample_rate: 16_000,
            n_fft: 512,
            hop_length: 256,
            n_mels: 40,
            fmin: 0.0,
            fmax: 8000.0,
            max_audio_seconds: 1.0,
            power: 2.0,
            log_eps: 1.0e-10,
        };
        let pre = AudioPreprocessor::new(config).unwrap();
        let tensor = pre.preprocess_path(&temp, &Device::Cpu).unwrap();
        assert_eq!(tensor.dims(), &[40, 61]);
        let _ = std::fs::remove_file(&temp);
    }

    #[test]
    fn preprocess_bytes_handles_stereo_float() {
        // Build a minimal 32-bit float stereo WAV.
        let sample_rate = 16_000u32;
        let sample_count = 4usize;
        let mut data = Vec::with_capacity(sample_count * 8);
        for idx in 0..sample_count {
            let value = (idx as f32) / 4.0;
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
        }
        let mut bytes = Vec::new();
        let riff_size = 12 + 24 + 8 + data.len() as u32;
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&riff_size.to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
        bytes.extend_from_slice(&2u16.to_le_bytes()); // stereo
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        let byte_rate = sample_rate * 8;
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&8u16.to_le_bytes()); // block align
        bytes.extend_from_slice(&32u16.to_le_bytes()); // bits per sample
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&data);
        let config = MelSpectrogramConfig {
            n_fft: 4,
            hop_length: 2,
            n_mels: 8,
            sample_rate,
            fmin: 0.0,
            fmax: 8000.0,
            max_audio_seconds: 0.001,
            power: 2.0,
            log_eps: 1.0e-10,
        };
        let pre = AudioPreprocessor::new(config).unwrap();
        let tensor = pre.preprocess_bytes(&bytes, &Device::Cpu).unwrap();
        assert!(tensor.dims().iter().product::<usize>() > 0);
    }

    #[test]
    fn rejects_non_power_of_two_fft() {
        let config = MelSpectrogramConfig {
            n_fft: 300,
            ..MelSpectrogramConfig::default()
        };
        assert!(config.validate().is_err());
    }
}
