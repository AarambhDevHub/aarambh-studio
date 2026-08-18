//! Document chunking policy — fixed-size token-based chunks with overlap.
//!
//! Chunks are produced by tokenizing the source text (via the project
//! [`TokenizerLike`](aarambh_studio_core::TokenizerLike) trait) and slicing the
//! resulting token stream into windows of `chunk_size` tokens, each advanced
//! by `chunk_size - overlap` tokens. Overlap guarantees that a fact spanning a
//! chunk boundary is still retrievable from at least one chunk, while every
//! chunk receives a distinct, monotonically-increasing id and a byte offset
//! into the source — so overlap never produces duplicate index entries.

use std::path::{Path, PathBuf};

use aarambh_studio_core::{AarambhError, Result, TokenizerLike};
use serde::{Deserialize, Serialize};

/// A single chunk of a source document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Chunk {
    /// Monotonic, zero-based chunk id within the corpus build.
    pub id: u64,
    /// Decoded chunk text.
    pub text: String,
    /// Source file path the chunk was produced from.
    pub source: PathBuf,
    /// Byte offset of the chunk's first token's text within the source.
    pub offset: usize,
    /// Number of tokens in this chunk.
    pub len: usize,
}

impl Chunk {
    /// Construct a chunk with the given fields.
    pub fn new(id: u64, text: String, source: PathBuf, offset: usize, len: usize) -> Self {
        Self {
            id,
            text,
            source,
            offset,
            len,
        }
    }
}

/// Configuration for fixed-size token-based chunking with overlap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkingConfig {
    /// Target number of tokens per chunk.
    pub chunk_size: usize,
    /// Number of tokens shared between consecutive chunks.
    pub overlap: usize,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            chunk_size: 256,
            overlap: 32,
        }
    }
}

impl ChunkingConfig {
    /// Create a chunking policy, validating that overlap is strictly less
    /// than `chunk_size` (otherwise chunks would never advance).
    pub fn new(chunk_size: usize, overlap: usize) -> Result<Self> {
        if chunk_size == 0 {
            return Err(AarambhError::Config(
                "chunk_size must be greater than zero".into(),
            ));
        }
        if overlap >= chunk_size {
            return Err(AarambhError::Config(format!(
                "overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
            )));
        }
        Ok(Self {
            chunk_size,
            overlap,
        })
    }

    /// The number of tokens each window advances by.
    pub fn stride(&self) -> usize {
        self.chunk_size - self.overlap
    }
}

/// A chunker that splits tokenized text into overlapping windows.
#[derive(Debug, Clone)]
pub struct Chunker {
    config: ChunkingConfig,
}

impl Chunker {
    /// Construct a chunker from a validated policy.
    pub fn new(config: ChunkingConfig) -> Self {
        Self { config }
    }

    /// Return the chunking policy.
    pub fn config(&self) -> &ChunkingConfig {
        &self.config
    }

    /// Chunk a single text into overlapping token windows.
    ///
    /// Each chunk's `offset` is the byte offset of its first token's decoded
    /// text within `text`, computed by decoding the prefix up to that token.
    /// This keeps offsets honest without depending on tokenizer offset mapping.
    pub fn chunk_text<T: TokenizerLike>(
        &self,
        tokenizer: &T,
        text: &str,
        source: &Path,
        start_id: u64,
    ) -> Result<Vec<Chunk>> {
        let token_ids = tokenizer.encode(text)?;
        if token_ids.is_empty() {
            return Ok(Vec::new());
        }
        let stride = self.config.stride();
        let chunk_size = self.config.chunk_size;
        let total = token_ids.len();

        let mut chunks = Vec::new();
        let mut id = start_id;
        let mut start = 0usize;
        while start < total {
            let end = (start + chunk_size).min(total);
            let window = &token_ids[start..end];
            let decoded = tokenizer.decode(window)?;
            // Byte offset of the window's first token: decode the prefix
            // [0..start) and measure its length. For start == 0 this is 0.
            let offset = if start == 0 {
                0
            } else {
                let prefix = tokenizer.decode(&token_ids[..start])?;
                locate_prefix_offset(text, &prefix)
            };
            chunks.push(Chunk {
                id,
                text: decoded,
                source: source.to_path_buf(),
                offset,
                len: window.len(),
            });
            id += 1;
            if end == total {
                break;
            }
            start += stride;
        }
        Ok(chunks)
    }

    /// Chunk every `.txt`/`.md`/`.jsonl` file under a corpus directory,
    /// assigning monotonically-increasing chunk ids across files.
    pub fn chunk_corpus<T: TokenizerLike>(
        &self,
        tokenizer: &T,
        corpus_dir: &Path,
    ) -> Result<Vec<Chunk>> {
        let mut chunks = Vec::new();
        let mut next_id: u64 = 0;
        let mut files = collect_corpus_files(corpus_dir)?;
        files.sort();
        for file in files {
            let text = std::fs::read_to_string(&file).map_err(AarambhError::from)?;
            let file_chunks = self.chunk_text(tokenizer, &text, &file, next_id)?;
            next_id = next_id.saturating_add(file_chunks.len() as u64);
            chunks.extend(file_chunks);
        }
        Ok(chunks)
    }
}

/// Collect corpus files (`.txt`, `.md`, `.jsonl`) under `dir`, recursively.
fn collect_corpus_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !dir.is_dir() {
        return Err(AarambhError::Config(format!(
            "corpus path is not a directory: {}",
            dir.display()
        )));
    }
    visit_corpus(dir, &mut out)?;
    if out.is_empty() {
        return Err(AarambhError::Config(format!(
            "corpus directory {} contains no .txt/.md/.jsonl files",
            dir.display()
        )));
    }
    Ok(out)
}

fn visit_corpus(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir).map_err(AarambhError::from)? {
        let entry = entry.map_err(AarambhError::from)?;
        let path = entry.path();
        if path.is_dir() {
            visit_corpus(&path, out)?;
        } else if is_text_corpus_file(&path) {
            out.push(path);
        }
    }
    Ok(())
}

fn is_text_corpus_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("txt" | "md" | "jsonl")
    )
}

/// Locate where `prefix` begins inside `text`, returning a byte offset.
///
/// Tokenizer round-trips may normalize whitespace, so an exact substring
/// search can fail. This falls back to the byte length of `prefix` clamped to
/// `text.len()` — an honest best-effort offset. For plain ASCII corpora the
/// decoded prefix is a contiguous substring of the source and the offset is
/// exact.
fn locate_prefix_offset(text: &str, prefix: &str) -> usize {
    if prefix.is_empty() {
        return 0;
    }
    if let Some(idx) = text.find(prefix) {
        return idx + prefix.len();
    }
    // Whitespace-insensitive fallback: advance one byte at a time over `text`,
    // skipping ASCII whitespace in both, until `prefix` is exhausted.
    let mut ti = text.bytes();
    let mut pi = prefix.bytes();
    let mut consumed = 0usize;
    let skip_ws = |it: &mut std::str::Bytes| {
        let mut advanced = false;
        while let Some(b) = it.clone().next() {
            if b.is_ascii_whitespace() {
                it.next();
                advanced = true;
            } else {
                break;
            }
        }
        advanced
    };
    loop {
        skip_ws(&mut ti);
        skip_ws(&mut pi);
        match pi.next() {
            None => return consumed,
            Some(_) => match ti.next() {
                None => return consumed,
                Some(_) => consumed += 1,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stub character-level tokenizer for self-contained unit tests.
    struct StubTokenizer {
        vocab: std::collections::HashMap<String, u32>,
    }

    impl StubTokenizer {
        fn new() -> Self {
            // Character-level vocab for ASCII letters, digits, space, and
            // punctuation — enough to chunk deterministic text.
            let mut vocab = std::collections::HashMap::new();
            let chars: Vec<char> =
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-_\n'\""
                    .chars()
                    .collect();
            for (i, ch) in chars.iter().enumerate() {
                vocab.insert(ch.to_string(), i as u32);
            }
            Self { vocab }
        }
    }

    impl TokenizerLike for StubTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            let mut ids = Vec::new();
            for ch in text.chars() {
                match self.vocab.get(&ch.to_string()) {
                    Some(id) => ids.push(*id),
                    None => {
                        return Err(AarambhError::Tokenizer(format!(
                            "stub tokenizer has no id for character {ch:?}"
                        )));
                    }
                }
            }
            Ok(ids)
        }
        fn decode(&self, ids: &[u32]) -> Result<String> {
            let inv: std::collections::HashMap<u32, String> =
                self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
            let mut out = String::new();
            for id in ids {
                match inv.get(id) {
                    Some(s) => out.push_str(s),
                    None => {
                        return Err(AarambhError::Tokenizer(format!(
                            "stub tokenizer has no token for id {id}"
                        )));
                    }
                }
            }
            Ok(out)
        }
        fn vocab_size(&self) -> usize {
            self.vocab.len()
        }
        fn eos_token_id(&self) -> u32 {
            0
        }
        fn bos_token_id(&self) -> Option<u32> {
            None
        }
    }

    #[test]
    fn overlap_must_be_strictly_less_than_chunk_size() {
        assert!(ChunkingConfig::new(0, 0).is_err());
        assert!(ChunkingConfig::new(64, 64).is_err());
        assert!(ChunkingConfig::new(64, 65).is_err());
        let ok = ChunkingConfig::new(64, 32).unwrap();
        assert_eq!(ok.stride(), 32);
    }

    #[test]
    fn chunking_with_overlap_does_not_duplicate_index_entries_incorrectly() {
        let tok = StubTokenizer::new();
        let config = ChunkingConfig::new(8, 4).unwrap();
        let chunker = Chunker::new(config);
        // 24 distinct characters => 24 tokens; stride 4 => windows at
        // 0,4,8,12,16,20. Each window is 8 distinct chars so consecutive
        // chunks share exactly the 4-char overlap but are not identical.
        let text = "abcdefghijklmnopqrstuvwx";
        let chunks = chunker
            .chunk_text(&tok, text, Path::new("doc.txt"), 0)
            .unwrap();
        // Each chunk has a distinct, monotonically-increasing id.
        let ids: Vec<u64> = chunks.iter().map(|c| c.id).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted, "ids must be monotonic");
        assert_eq!(
            ids.iter().collect::<std::collections::HashSet<_>>().len(),
            ids.len(),
            "no duplicate ids"
        );
        // Each chunk is 8 chars; consecutive chunks share exactly the overlap.
        for win in chunks.windows(2) {
            let a = &win[0].text;
            let b = &win[1].text;
            assert_eq!(a.len(), 8, "non-final chunks are full-size");
            let tail = &a[a.len() - 4..];
            let head = &b[..4];
            assert_eq!(
                tail, head,
                "consecutive chunks must share the overlap window"
            );
            // Because the source is a string of distinct characters, two
            // chunks starting at different offsets are different text —
            // overlap shares a window but never duplicates the whole chunk.
            assert_ne!(win[0].text, win[1].text);
        }
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        let tok = StubTokenizer::new();
        let chunker = Chunker::new(ChunkingConfig::new(8, 4).unwrap());
        let chunks = chunker
            .chunk_text(&tok, "", Path::new("empty.txt"), 0)
            .unwrap();
        assert!(chunks.is_empty());
    }

    #[test]
    fn corpus_dir_without_text_files_errors() {
        let tok = StubTokenizer::new();
        let chunker = Chunker::new(ChunkingConfig::new(8, 4).unwrap());
        let tmp = tempdir();
        // Add a non-text file so the dir is non-empty but has no corpus files.
        std::fs::write(tmp.join("image.bin"), b"\x00\x01").unwrap();
        let err = chunker.chunk_corpus(&tok, &tmp).unwrap_err();
        assert!(matches!(err, AarambhError::Config(_)), "{err:?}");
    }

    #[test]
    fn corpus_chunking_assigns_monotonic_ids_across_files() {
        let tok = StubTokenizer::new();
        let chunker = Chunker::new(ChunkingConfig::new(6, 2).unwrap());
        let tmp = tempdir();
        std::fs::write(tmp.join("a.txt"), "aaaaaaaaaaaaaaaa").unwrap();
        std::fs::write(tmp.join("b.txt"), "bbbbbbbbbbbbbbbb").unwrap();
        let chunks = chunker.chunk_corpus(&tok, &tmp).unwrap();
        let ids: Vec<u64> = chunks.iter().map(|c| c.id).collect();
        let expected: Vec<u64> = (0..ids.len() as u64).collect();
        assert_eq!(ids, expected);
        // sanity: each chunk's source is one of the two files
        for c in &chunks {
            let name = c.source.file_name().unwrap().to_string_lossy().to_string();
            assert!(name == "a.txt" || name == "b.txt", "{name}");
        }
    }

    /// Tiny tempdir helper (avoids pulling a dev-dependency for one test).
    fn tempdir() -> PathBuf {
        let mut dir = std::env::temp_dir();
        let nano = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        dir.push(format!("aarambh-rag-test-{}", nano));
        std::fs::create_dir_all(&dir).unwrap();
        // Clean up on process exit best-effort.
        let to_remove = dir.clone();
        std::thread::spawn(move || {
            // Best-effort: tests are short-lived; rely on tmp reaper.
            let _ = std::fs::remove_dir_all(&to_remove);
        });
        dir
    }
}
