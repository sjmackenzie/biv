//! Find GGUF models in the Hugging Face cache, where skelm downloads them.
//!
//! Names follow skelm's convention: `hf.co/<owner>/<repo>[:<quant>]`, the
//! `hf.co/` prefix being optional.

use std::path::{Path, PathBuf};

/// The hub cache directory, as `huggingface_hub` (and skelm) resolve it
pub fn cache_dir() -> PathBuf {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    if let Some(dir) = var("HF_HUB_CACHE").or_else(|| var("HUGGINGFACE_HUB_CACHE")) {
        return PathBuf::from(dir);
    }
    let hf_home = var("HF_HOME").map(PathBuf::from).unwrap_or_else(|| {
        var("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::home_dir().unwrap_or_default().join(".cache"))
            .join("huggingface")
    });
    hf_home.join("hub")
}

fn gguf_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            gguf_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "gguf") {
            out.push(p);
        }
    }
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Whether a file is the first (or only) shard of a model, and not a projector
fn is_primary(p: &Path) -> bool {
    let name = file_name(p).to_ascii_lowercase();
    if name.starts_with("mmproj") {
        return false;
    }
    !name.contains("-of-") || name.contains("-00001-of-")
}

/// The SHA-256 of a model of the cache, from the names of its blobs: the
/// file's, or for a split model the XOR of its shards'. None when a shard
/// is not in the cache.
pub fn model_sha256(path: &Path) -> Option<[u8; 32]> {
    let mut sum = [0; 32];
    for shard in crate::shard_paths(path) {
        for (a, b) in sum.iter_mut().zip(blob_sha256(&shard)?) {
            *a ^= b;
        }
    }
    Some(sum)
}

/// The SHA-256 of a file of the cache: its blob is named after it
fn blob_sha256(path: &Path) -> Option<[u8; 32]> {
    let real = std::fs::canonicalize(path).ok()?;
    if real.parent()?.file_name()? != "blobs" {
        return None;
    }
    let name = real.file_name()?.to_str()?;
    if name.len() != 64 {
        return None;
    }
    let mut sum = [0; 32];
    for (b, pair) in sum.iter_mut().zip(name.as_bytes().chunks(2)) {
        *b = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(sum)
}

/// Resolve a model reference to a local GGUF path
pub fn resolve(model: &str) -> Result<PathBuf, String> {
    let as_path = Path::new(model);
    if as_path.is_file() {
        return Ok(as_path.to_path_buf());
    }
    let name = ["hf.co/", "huggingface.co/"]
        .iter()
        .find_map(|p| model.strip_prefix(p))
        .unwrap_or(model);
    let (repo, quant) = match name.rsplit_once(':') {
        Some((r, q)) => (r, Some(q)),
        None => (name, None),
    };
    if repo.split('/').count() != 2 {
        return Err(format!(
            "'{model}' is neither a file nor a '<owner>/<repo>[:<quant>]' name"
        ));
    }

    let repo_dir = cache_dir().join(format!("models--{}", repo.replace('/', "--")));
    let commit = std::fs::read_to_string(repo_dir.join("refs/main"))
        .map(|c| c.trim().to_string())
        .ok();
    let snapshots = repo_dir.join("snapshots");
    let snapshot = match commit {
        Some(c) if snapshots.join(&c).is_dir() => snapshots.join(c),
        _ => std::fs::read_dir(&snapshots)
            .ok()
            .and_then(|mut d| d.find_map(|e| e.ok().map(|e| e.path())))
            .ok_or_else(|| format!("{repo} is not in the cache ({})", repo_dir.display()))?,
    };

    let mut files = Vec::new();
    gguf_files(&snapshot, &mut files);
    files.retain(|p| is_primary(p));
    files.sort();
    if let Some(q) = quant {
        let q = q.to_ascii_lowercase();
        // match the quant as a whole name component: Q4_1 must not pick Q4_1_XL
        files.retain(|p| {
            let n = file_name(p).to_ascii_lowercase();
            let stem = n.strip_suffix(".gguf").unwrap_or(&n);
            let stem = stem.split("-00001-of-").next().unwrap_or(stem);
            stem.ends_with(&format!("-{q}")) || stem.ends_with(&format!(".{q}")) || stem == q
        });
    }
    match files.len() {
        0 => Err(format!(
            "no GGUF matching '{model}' in {}",
            snapshot.display()
        )),
        1 => Ok(files.remove(0)),
        _ => {
            let names: Vec<String> = files.iter().map(|p| file_name(p)).collect();
            Err(format!(
                "'{model}' is ambiguous, add a quant (e.g. :Q4_1): {}",
                names.join(", ")
            ))
        }
    }
}
