//! content-addressed store: bytes live at `<upload_dir>/<ab>/<sha256>`, the db only holds the relative path

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct Storage {
    root: PathBuf,
}

pub struct Stored {
    pub sha256: String,
    pub relative_path: String,
    pub byte_size: usize,
    pub is_new: bool,
}

impl Storage {
    pub async fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        tokio::fs::create_dir_all(&root).await?;
        Ok(Storage { root })
    }

    pub fn absolute(&self, relative: &str) -> PathBuf {
        self.root.join(relative)
    }

    pub async fn read(&self, relative: &str) -> std::io::Result<Vec<u8>> {
        tokio::fs::read(self.absolute(relative)).await
    }

    pub async fn read_head(&self, relative: &str, limit: usize) -> std::io::Result<Vec<u8>> {
        use tokio::io::AsyncReadExt;

        let file = tokio::fs::File::open(self.absolute(relative)).await?;
        let mut head = Vec::with_capacity(limit.min(64 * 1024));
        file.take(limit as u64).read_to_end(&mut head).await?;
        Ok(head)
    }

    pub async fn put(&self, bytes: &[u8]) -> std::io::Result<Stored> {
        let sha256 = hex(&Sha256::digest(bytes));
        let relative = format!("{}/{}", &sha256[..2], sha256);
        let path = self.root.join(&relative);

        let exists = tokio::fs::try_exists(&path).await.unwrap_or(false);
        if !exists {
            if let Some(parent) = path.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&path, bytes).await?;
        }

        Ok(Stored {
            sha256,
            relative_path: relative,
            byte_size: bytes.len(),
            is_new: !exists,
        })
    }

    /// caller counts refs; the blob goes only at zero
    pub async fn remove_if_unreferenced(
        &self,
        relative: &str,
        remaining_refs: i64,
    ) -> std::io::Result<()> {
        if remaining_refs == 0 {
            match tokio::fs::remove_file(self.absolute(relative)).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    })
}
