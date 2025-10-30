use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use tempfile::TempDir;
use crate::types::ProcessError;

pub struct BinaryExtractor {
    _temp_dir: TempDir,
    temp_path: PathBuf,
    binaries: HashMap<String, PathBuf>,
}

impl BinaryExtractor {
    pub fn new() -> Result<Self, ProcessError> {
        let temp_dir = TempDir::new()
            .map_err(|e| ProcessError::ExtractionFailed(format!("Failed to create temp dir: {}", e)))?;
        let temp_path = temp_dir.path().to_path_buf();

        log::info!("Created temporary directory: {}", temp_path.display());

        Ok(Self {
            _temp_dir: temp_dir,
            temp_path,
            binaries: HashMap::new(),
        })
    }

    /// Force cleanup of temporary directory
    pub fn cleanup(&self) {
        log::info!("Cleaning up temporary directory: {}", self.temp_path.display());
        // The TempDir will be automatically cleaned up when dropped
        // This method is just for explicit logging
    }
    
    pub fn add_binary(&mut self, name: &str, binary_data: &[u8]) -> Result<(), ProcessError> {
        let binary_path = self.temp_path.join(name);
        self.extract_binary(&binary_path, binary_data, name)?;
        self.binaries.insert(name.to_string(), binary_path);
        Ok(())
    }
    
    pub fn add_multiple_binaries(&mut self, binaries: &[(&str, &[u8])]) -> Result<(), ProcessError> {
        for (name, data) in binaries {
            self.add_binary(name, data)?;
        }
        Ok(())
    }
    
    fn extract_binary(
        &self,
        path: &Path,
        binary_data: &[u8],
        name: &str,
    ) -> Result<(), ProcessError> {
        {
            let mut file = fs::File::create(path)?;
            file.write_all(binary_data)?;
            file.flush()?;
        }
        
        // Make the binary executable
        let mut perms = fs::metadata(path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(path, perms)?;
        
        log::info!("Extracted {} binary to: {}", name, path.display());
        
        Ok(())
    }
    
    pub fn get_binary_path(&self, name: &str) -> Option<&Path> {
        self.binaries.get(name).map(|p| p.as_path())
    }
    
    pub fn list_binaries(&self) -> Vec<&str> {
        self.binaries.keys().map(|s| s.as_str()).collect()
    }
    
    pub fn temp_dir(&self) -> &Path {
        &self.temp_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_binary_extractor_creation() {
        let extractor = BinaryExtractor::new().unwrap();
        assert!(extractor.temp_dir().exists());
        assert_eq!(extractor.list_binaries().len(), 0);
    }
    
    #[test]
    fn test_add_binary() {
        let mut extractor = BinaryExtractor::new().unwrap();
        let test_data = b"#!/bin/sh\necho 'test'";
        
        extractor.add_binary("test_script", test_data).unwrap();
        
        assert_eq!(extractor.list_binaries().len(), 1);
        assert!(extractor.get_binary_path("test_script").is_some());
        
        let path = extractor.get_binary_path("test_script").unwrap();
        assert!(path.exists());
        
        // Check if file is executable
        let metadata = fs::metadata(path).unwrap();
        let permissions = metadata.permissions();
        assert!(permissions.mode() & 0o111 != 0);
    }
    
    #[test]
    fn test_add_multiple_binaries() {
        let mut extractor = BinaryExtractor::new().unwrap();
        
        let binaries = [
            ("script1", b"#!/bin/sh\necho '1'" as &[u8]),
            ("script2", b"#!/bin/sh\necho '2'" as &[u8]),
            ("script3", b"#!/bin/sh\necho '3'" as &[u8]),
        ];
        
        extractor.add_multiple_binaries(&binaries).unwrap();
        
        assert_eq!(extractor.list_binaries().len(), 3);
        for (name, _) in &binaries {
            assert!(extractor.get_binary_path(name).is_some());
        }
    }
}