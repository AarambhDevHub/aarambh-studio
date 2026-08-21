use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use aarambh_studio_core::{AarambhError, Result};
use candle_core::Tensor;
use hayro::{InterpreterSettings, Pdf, RenderSettings, render};
use image::{ImageReader, RgbImage};
use serde::{Deserialize, Serialize};

/// PDF rasterization limits used by document training and inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(default)]
pub struct PageRasterizerConfig {
    /// Target PDF rendering resolution in dots per inch.
    pub target_dpi: u32,
    /// Maximum number of pages returned for one document.
    pub max_pages_per_document: usize,
    /// Maximum decoded or rendered pixels accepted for one page.
    pub max_page_pixels: usize,
}

impl Default for PageRasterizerConfig {
    fn default() -> Self {
        Self {
            target_dpi: 150,
            max_pages_per_document: 16,
            max_page_pixels: 32_000_000,
        }
    }
}

impl PageRasterizerConfig {
    /// Validate page rendering limits.
    pub fn validate(&self) -> Result<()> {
        if self.target_dpi == 0 || self.max_pages_per_document == 0 || self.max_page_pixels == 0 {
            return Err(AarambhError::Config(
                "document target_dpi, max_pages_per_document, and max_page_pixels must be non-zero"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// A document represented by one PDF/image file or an ordered list of page images.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DocumentSource {
    /// PDF or single raster-image path.
    File(PathBuf),
    /// Ordered raster-image paths for a multi-page scanned document.
    PageImages(Vec<PathBuf>),
}

impl DocumentSource {
    /// Return the source paths in deterministic page order.
    pub fn paths(&self) -> &[PathBuf] {
        match self {
            Self::File(path) => std::slice::from_ref(path),
            Self::PageImages(paths) => paths,
        }
    }

    fn validate(&self) -> Result<()> {
        match self {
            Self::File(path) if path.as_os_str().is_empty() => Err(AarambhError::Config(
                "document file path must not be empty".into(),
            )),
            Self::PageImages(paths) if paths.is_empty() => Err(AarambhError::Config(
                "document page image list must not be empty".into(),
            )),
            Self::PageImages(paths) if paths.iter().any(|path| path.as_os_str().is_empty()) => Err(
                AarambhError::Config("document page image paths must not be empty".into()),
            ),
            _ => Ok(()),
        }
    }
}

/// One rendered document page and its 1-based source page number.
#[derive(Debug, Clone)]
pub struct RasterizedPage {
    /// 1-based page number in the source document.
    pub page_number: usize,
    /// Decoded RGB page pixels.
    pub image: RgbImage,
}

/// Selected pages produced from one document source.
#[derive(Debug, Clone)]
pub struct RasterizedDocument {
    /// Number of pages in the original source.
    pub source_page_count: usize,
    /// Whether implicit first-page selection omitted pages due to the configured limit.
    pub truncated: bool,
    /// Rendered pages in the requested order.
    pub pages: Vec<RasterizedPage>,
}

/// Resource-bounded PDF and scanned-page rasterizer.
#[derive(Debug, Clone)]
pub struct PageRasterizer {
    config: PageRasterizerConfig,
}

impl PageRasterizer {
    /// Create a page rasterizer from explicit limits.
    pub fn new(config: PageRasterizerConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Return the rasterizer configuration.
    pub fn config(&self) -> &PageRasterizerConfig {
        &self.config
    }

    /// Render selected 1-based pages, or the first configured page window when omitted.
    pub fn rasterize(
        &self,
        source: &DocumentSource,
        selected_pages: Option<&[usize]>,
    ) -> Result<RasterizedDocument> {
        source.validate()?;
        match source {
            DocumentSource::File(path) if is_pdf(path) => self.rasterize_pdf(path, selected_pages),
            DocumentSource::File(path) => {
                self.rasterize_images(std::slice::from_ref(path), selected_pages)
            }
            DocumentSource::PageImages(paths) => self.rasterize_images(paths, selected_pages),
        }
    }

    fn rasterize_pdf(
        &self,
        path: &Path,
        selected_pages: Option<&[usize]>,
    ) -> Result<RasterizedDocument> {
        let bytes = std::fs::read(path).map_err(|error| {
            AarambhError::Io(std::io::Error::new(
                error.kind(),
                format!("failed to read PDF {}: {error}", path.display()),
            ))
        })?;
        let pdf = Pdf::new(Arc::new(bytes)).map_err(|error| {
            AarambhError::Config(format!("failed to parse PDF {}: {error:?}", path.display()))
        })?;
        let source_page_count = pdf.pages().len();
        let (indices, truncated) = selected_indices(
            source_page_count,
            selected_pages,
            self.config.max_pages_per_document,
        )?;
        let scale = self.config.target_dpi as f32 / 72.0;
        let settings = InterpreterSettings::default();
        let mut pages = Vec::with_capacity(indices.len());
        for index in indices {
            let page = &pdf.pages()[index];
            let (width_points, height_points) = page.render_dimensions();
            let width = checked_render_extent(width_points, scale, "width")?;
            let height = checked_render_extent(height_points, scale, "height")?;
            self.validate_pixels(width as usize, height as usize, path, index + 1)?;
            let pixmap = render(
                page,
                &settings,
                &RenderSettings {
                    x_scale: scale,
                    y_scale: scale,
                    width: Some(width),
                    height: Some(height),
                },
            );
            let rgba = pixmap.take_u8();
            let image = rgba_to_rgb(rgba, width as u32, height as u32)?;
            pages.push(RasterizedPage {
                page_number: index + 1,
                image,
            });
        }
        Ok(RasterizedDocument {
            source_page_count,
            truncated,
            pages,
        })
    }

    fn rasterize_images(
        &self,
        paths: &[PathBuf],
        selected_pages: Option<&[usize]>,
    ) -> Result<RasterizedDocument> {
        let source_page_count = paths.len();
        let (indices, truncated) = selected_indices(
            source_page_count,
            selected_pages,
            self.config.max_pages_per_document,
        )?;
        let mut pages = Vec::with_capacity(indices.len());
        for index in indices {
            let path = &paths[index];
            let (width, height) = image::image_dimensions(path).map_err(|error| {
                AarambhError::Config(format!(
                    "failed to inspect document page {}: {error}",
                    path.display()
                ))
            })?;
            self.validate_pixels(width as usize, height as usize, path, index + 1)?;
            let reader = ImageReader::open(path).map_err(|error| {
                AarambhError::Io(std::io::Error::new(
                    error.kind(),
                    format!("failed to open document page {}: {error}", path.display()),
                ))
            })?;
            let image = reader.decode().map_err(|error| {
                AarambhError::Config(format!(
                    "failed to decode document page {}: {error}",
                    path.display()
                ))
            })?;
            pages.push(RasterizedPage {
                page_number: index + 1,
                image: image.to_rgb8(),
            });
        }
        Ok(RasterizedDocument {
            source_page_count,
            truncated,
            pages,
        })
    }

    fn validate_pixels(&self, width: usize, height: usize, path: &Path, page: usize) -> Result<()> {
        let pixels = width.checked_mul(height).ok_or_else(|| {
            AarambhError::Config(format!(
                "document page dimensions overflow for {} page {page}",
                path.display()
            ))
        })?;
        if width == 0 || height == 0 || pixels > self.config.max_page_pixels {
            return Err(AarambhError::Config(format!(
                "document page {} page {page} is {width}x{height} ({pixels} pixels), limit is {}",
                path.display(),
                self.config.max_page_pixels
            )));
        }
        Ok(())
    }
}

impl Default for PageRasterizer {
    fn default() -> Self {
        Self::new(PageRasterizerConfig::default()).expect("valid page rasterizer defaults")
    }
}

fn is_pdf(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("pdf"))
}

fn checked_render_extent(points: f32, scale: f32, axis: &str) -> Result<u16> {
    let pixels = (points * scale).ceil();
    if !pixels.is_finite() || pixels < 1.0 || pixels > u16::MAX as f32 {
        return Err(AarambhError::Config(format!(
            "rendered PDF page {axis} {pixels} is outside 1..={}",
            u16::MAX
        )));
    }
    Ok(pixels as u16)
}

fn selected_indices(
    total: usize,
    selected_pages: Option<&[usize]>,
    limit: usize,
) -> Result<(Vec<usize>, bool)> {
    if total == 0 {
        return Err(AarambhError::Config("document contains no pages".into()));
    }
    let Some(selected) = selected_pages else {
        return Ok(((0..total.min(limit)).collect(), total > limit));
    };
    if selected.is_empty() {
        return Err(AarambhError::Config(
            "document page selection must not be empty".into(),
        ));
    }
    if selected.len() > limit {
        return Err(AarambhError::Config(format!(
            "selected {} document pages, limit is {limit}",
            selected.len()
        )));
    }
    let mut seen = HashSet::with_capacity(selected.len());
    let mut indices = Vec::with_capacity(selected.len());
    for &page in selected {
        if page == 0 || page > total {
            return Err(AarambhError::Config(format!(
                "document page {page} is outside 1..={total}"
            )));
        }
        if !seen.insert(page) {
            return Err(AarambhError::Config(format!(
                "document page {page} was selected more than once"
            )));
        }
        indices.push(page - 1);
    }
    Ok((indices, false))
}

fn rgba_to_rgb(rgba: Vec<u8>, width: u32, height: u32) -> Result<RgbImage> {
    let pixels = width as usize * height as usize;
    if rgba.len() != pixels * 4 {
        return Err(AarambhError::Shape(format!(
            "PDF renderer returned {} RGBA bytes for {width}x{height}",
            rgba.len()
        )));
    }
    let mut rgb = Vec::with_capacity(pixels * 3);
    for pixel in rgba.as_chunks::<4>().0 {
        rgb.extend_from_slice(&pixel[..3]);
    }
    RgbImage::from_raw(width, height, rgb)
        .ok_or_else(|| AarambhError::Shape("failed to construct rendered RGB page".into()))
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SourceFileFingerprint {
    path: PathBuf,
    modified_nanos: u128,
    file_size: u64,
}

/// Key for detached frozen-encoder document features.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DocumentFeatureCacheKey {
    sources: Vec<SourceFileFingerprint>,
    selected_pages: Option<Vec<usize>>,
    rasterizer: PageRasterizerConfig,
    encoder_signature: String,
}

impl DocumentFeatureCacheKey {
    /// Build a cache key from source metadata and preprocessing configuration.
    pub fn new(
        source: &DocumentSource,
        selected_pages: Option<&[usize]>,
        rasterizer: PageRasterizerConfig,
        encoder_signature: impl Into<String>,
    ) -> Result<Self> {
        let mut sources = Vec::with_capacity(source.paths().len());
        for path in source.paths() {
            let path = path.canonicalize()?;
            let metadata = path.metadata()?;
            let modified_nanos = metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                .map_or(0, |duration| duration.as_nanos());
            sources.push(SourceFileFingerprint {
                path,
                modified_nanos,
                file_size: metadata.len(),
            });
        }
        Ok(Self {
            sources,
            selected_pages: selected_pages.map(<[usize]>::to_vec),
            rasterizer,
            encoder_signature: encoder_signature.into(),
        })
    }
}

/// Bounded FIFO cache for detached, pre-projector document page features.
#[derive(Debug)]
pub struct DocumentFeatureCache {
    capacity: usize,
    entries: HashMap<DocumentFeatureCacheKey, Tensor>,
    order: VecDeque<DocumentFeatureCacheKey>,
}

impl DocumentFeatureCache {
    /// Create a cache with the given maximum number of documents.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            order: VecDeque::with_capacity(capacity),
        }
    }

    /// Return cached frozen-encoder features when present.
    pub fn get(&self, key: &DocumentFeatureCacheKey) -> Option<Tensor> {
        self.entries.get(key).cloned()
    }

    /// Insert detached frozen features and evict the oldest key when full.
    pub fn insert(&mut self, key: DocumentFeatureCacheKey, features: Tensor) {
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

    /// Return the number of cached documents.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the cache contains no documents.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn page_selection_is_ordered_and_one_based() {
        let (indices, truncated) = selected_indices(5, Some(&[4, 2]), 4).unwrap();
        assert_eq!(indices, vec![3, 1]);
        assert!(!truncated);
    }

    #[test]
    fn implicit_selection_reports_truncation() {
        let (indices, truncated) = selected_indices(5, None, 2).unwrap();
        assert_eq!(indices, vec![0, 1]);
        assert!(truncated);
    }

    #[test]
    fn duplicate_pages_are_rejected() {
        assert!(selected_indices(5, Some(&[2, 2]), 4).is_err());
    }

    #[test]
    fn native_pdf_rasterizer_renders_selected_pages() {
        let path =
            std::env::temp_dir().join(format!("aarambh_phase36_pdf_{}.pdf", std::process::id()));
        std::fs::write(&path, two_page_pdf()).unwrap();
        let rasterizer = PageRasterizer::new(PageRasterizerConfig {
            target_dpi: 72,
            max_pages_per_document: 2,
            max_page_pixels: 128 * 128,
        })
        .unwrap();
        let rendered = rasterizer
            .rasterize(&DocumentSource::File(path.clone()), Some(&[2, 1]))
            .unwrap();
        let _ = std::fs::remove_file(path);
        assert_eq!(rendered.source_page_count, 2);
        assert_eq!(
            rendered
                .pages
                .iter()
                .map(|page| page.page_number)
                .collect::<Vec<_>>(),
            vec![2, 1]
        );
        assert!(
            rendered
                .pages
                .iter()
                .all(|page| { page.image.width() == 64 && page.image.height() == 48 })
        );
    }

    fn two_page_pdf() -> Vec<u8> {
        let page_one = b"1 0 0 rg 0 0 64 48 re f\n";
        let page_two = b"0 0 1 rg 0 0 64 48 re f\n";
        let objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 48] /Resources << >> /Contents 4 0 R >>".to_vec(),
            stream_object(page_one),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 64 48] /Resources << >> /Contents 6 0 R >>".to_vec(),
            stream_object(page_two),
        ];
        let mut pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n".to_vec();
        let mut offsets = vec![0usize];
        for (index, object) in objects.iter().enumerate() {
            offsets.push(pdf.len());
            writeln!(pdf, "{} 0 obj", index + 1).unwrap();
            pdf.extend_from_slice(object);
            pdf.extend_from_slice(b"\nendobj\n");
        }
        let xref = pdf.len();
        write!(pdf, "xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).unwrap();
        for offset in offsets.iter().skip(1) {
            writeln!(pdf, "{offset:010} 00000 n ").unwrap();
        }
        write!(
            pdf,
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .unwrap();
        pdf
    }

    fn stream_object(stream: &[u8]) -> Vec<u8> {
        let mut object = format!("<< /Length {} >>\nstream\n", stream.len()).into_bytes();
        object.extend_from_slice(stream);
        object.extend_from_slice(b"endstream");
        object
    }
}
