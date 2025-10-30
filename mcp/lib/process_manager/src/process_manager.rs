use crate::binary_extractor::BinaryExtractor;
use crate::process_runner::{ProcessRunner, OutputStream};
use crate::types::{ProcessConfig, ProcessError, ProcessInfo};
use dashmap::DashMap;
use std::sync::Arc;
use uuid::Uuid;

pub struct ProcessManager {
    binary_extractor: Arc<BinaryExtractor>,
    processes: Arc<DashMap<Uuid, ProcessRunner>>,
}

impl ProcessManager {
    pub fn new(binary_extractor: BinaryExtractor) -> Self {
        Self {
            binary_extractor: Arc::new(binary_extractor),
            processes: Arc::new(DashMap::new()),
        }
    }

    /// Force cleanup of temporary directories
    pub fn cleanup(&self) {
        log::info!("Cleaning up ProcessManager resources");
        self.binary_extractor.cleanup();
        // Stop all running processes
        let processes: Vec<Uuid> = self.processes.iter().map(|entry| *entry.key()).collect();
        for id in processes {
            if let Some((_, mut runner)) = self.processes.remove(&id) {
                tokio::spawn(async move {
                    let _ = runner.stop().await;
                });
            }
        }
    }
    
    pub async fn start_process(&self, config: ProcessConfig) -> Result<Uuid, ProcessError> {
        // Get binary path
        let binary_path = self.binary_extractor
            .get_binary_path(&config.binary_name)
            .ok_or_else(|| ProcessError::BinaryNotFound(config.binary_name.clone()))?;
        
        let process_name = config.name.clone();
        
        // Start the process
        let runner = ProcessRunner::start(
            binary_path.to_str().unwrap(),
            config.name,
            config.args,
        ).await?;
        
        let id = runner.id();
        self.processes.insert(id, runner);
        
        log::info!("Started process {} with ID {}", process_name, id);
        Ok(id)
    }
    
    pub async fn start_process_with_sudo(&self, config: ProcessConfig, sudo_password: &str) -> Result<Uuid, ProcessError> {
        // Get binary path
        let binary_path = self.binary_extractor
            .get_binary_path(&config.binary_name)
            .ok_or_else(|| ProcessError::BinaryNotFound(config.binary_name.clone()))?;
        
        let process_name = config.name.clone();
        
        // Start the process with sudo
        let runner = ProcessRunner::start_with_sudo(
            binary_path.to_str().unwrap(),
            config.name,
            config.args,
            sudo_password,
        ).await?;
        
        let id = runner.id();
        self.processes.insert(id, runner);
        
        log::info!("Started process {} with ID {} (using sudo)", process_name, id);
        Ok(id)
    }
    
    pub async fn start_multiple_processes(&self, configs: Vec<ProcessConfig>) -> Vec<Result<Uuid, ProcessError>> {
        let mut results = Vec::new();
        
        for config in configs {
            results.push(self.start_process(config).await);
        }
        
        results
    }
    
    pub async fn stop_process(&self, id: Uuid) -> Result<(), ProcessError> {
        match self.processes.get_mut(&id) {
            Some(mut runner) => {
                runner.stop().await?;
                log::info!("Stopped process {}", id);
                Ok(())
            }
            None => Err(ProcessError::ProcessNotFound(id)),
        }
    }
    
    pub async fn stop_all_processes(&self) -> Vec<(Uuid, Result<(), ProcessError>)> {
        let ids: Vec<Uuid> = self.processes.iter().map(|entry| *entry.key()).collect();
        let mut results = Vec::new();
        
        for id in ids {
            results.push((id, self.stop_process(id).await));
        }
        
        results
    }
    
    pub fn kill_process(&self, id: Uuid) -> Result<(), ProcessError> {
        match self.processes.remove(&id) {
            Some(_) => {
                log::info!("Killed process {} (force removed)", id);
                Ok(())
            }
            None => Err(ProcessError::ProcessNotFound(id)),
        }
    }
    
    pub fn list_processes(&self) -> Vec<ProcessInfo> {
        self.processes
            .iter()
            .map(|entry| entry.value().info().clone())
            .collect()
    }
    
    pub fn get_process_info(&self, id: Uuid) -> Option<ProcessInfo> {
        self.processes.get(&id).map(|runner| runner.info().clone())
    }
    
    pub fn get_output_stream(&self, id: Uuid) -> Option<OutputStream> {
        self.processes.get_mut(&id).and_then(|mut runner| runner.get_output_stream())
    }
    
    pub fn get_running_processes(&self) -> Vec<ProcessInfo> {
        self.processes
            .iter()
            .filter(|entry| entry.value().is_running())
            .map(|entry| entry.value().info().clone())
            .collect()
    }
    
    pub fn get_process_count(&self) -> usize {
        self.processes.len()
    }
    
    pub fn get_running_count(&self) -> usize {
        self.processes.iter().filter(|entry| entry.value().is_running()).count()
    }
    
    pub fn available_binaries(&self) -> Vec<&str> {
        self.binary_extractor.list_binaries()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary_extractor::BinaryExtractor;
    use futures::StreamExt;
    
    #[tokio::test]
    async fn test_process_manager_basic() {
        let mut extractor = BinaryExtractor::new().unwrap();
        
        // Add a simple test script
        let script = b"#!/bin/sh\necho 'Hello from test'\nsleep 0.1\necho 'Done'";
        extractor.add_binary("test_script", script).unwrap();
        
        let manager = ProcessManager::new(extractor);
        
        // Start a process
        let config = ProcessConfig {
            name: "test_process".to_string(),
            binary_name: "test_script".to_string(),
            args: vec![],
            env: Default::default(),
            working_dir: None,
        };
        
        let id = manager.start_process(config).await.unwrap();
        
        // Check process is running
        assert_eq!(manager.get_running_count(), 1);
        
        let info = manager.get_process_info(id).unwrap();
        assert_eq!(info.name, "test_process");
        assert_eq!(info.status, ProcessStatus::Running);
        
        // Get output stream
        if let Some(mut stream) = manager.get_output_stream(id) {
            let mut outputs = Vec::new();
            while let Some(line) = stream.next().await {
                outputs.push(line);
            }
            assert!(outputs.iter().any(|line| line.contains("Hello from test")));
            assert!(outputs.iter().any(|line| line.contains("Done")));
        }
        
        // Wait for process to complete
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }
    
    #[tokio::test]
    async fn test_process_manager_multiple() {
        let mut extractor = BinaryExtractor::new().unwrap();
        
        // Add multiple scripts
        extractor.add_multiple_binaries(&[
            ("script1", b"#!/bin/sh\necho 'Script 1'\nsleep 1"),
            ("script2", b"#!/bin/sh\necho 'Script 2'\nsleep 1"),
            ("script3", b"#!/bin/sh\necho 'Script 3'\nsleep 1"),
        ]).unwrap();
        
        let manager = ProcessManager::new(extractor);
        
        // Start multiple processes
        let configs = vec![
            ProcessConfig {
                name: "process1".to_string(),
                binary_name: "script1".to_string(),
                args: vec![],
                env: Default::default(),
                working_dir: None,
            },
            ProcessConfig {
                name: "process2".to_string(),
                binary_name: "script2".to_string(),
                args: vec![],
                env: Default::default(),
                working_dir: None,
            },
            ProcessConfig {
                name: "process3".to_string(),
                binary_name: "script3".to_string(),
                args: vec![],
                env: Default::default(),
                working_dir: None,
            },
        ];
        
        let results = manager.start_multiple_processes(configs).await;
        
        // All should succeed
        assert_eq!(results.len(), 3);
        for result in &results {
            assert!(result.is_ok());
        }
        
        // Check running count
        assert_eq!(manager.get_running_count(), 3);
        assert_eq!(manager.get_process_count(), 3);
        
        // Stop all processes
        let stop_results = manager.stop_all_processes().await;
        assert_eq!(stop_results.len(), 3);
        
        for (_, result) in stop_results {
            assert!(result.is_ok());
        }
    }
    
    #[tokio::test]
    async fn test_process_manager_stop_and_kill() {
        let mut extractor = BinaryExtractor::new().unwrap();
        extractor.add_binary("long_runner", b"#!/bin/sh\nwhile true; do sleep 1; done").unwrap();
        
        let manager = ProcessManager::new(extractor);
        
        // Start two processes
        let config1 = ProcessConfig {
            name: "process_to_stop".to_string(),
            binary_name: "long_runner".to_string(),
            args: vec![],
            env: Default::default(),
            working_dir: None,
        };
        
        let config2 = ProcessConfig {
            name: "process_to_kill".to_string(),
            binary_name: "long_runner".to_string(),
            args: vec![],
            env: Default::default(),
            working_dir: None,
        };
        
        let id1 = manager.start_process(config1).await.unwrap();
        let id2 = manager.start_process(config2).await.unwrap();
        
        assert_eq!(manager.get_running_count(), 2);
        
        // Stop one process gracefully
        manager.stop_process(id1).await.unwrap();
        
        // Kill the other process forcefully
        manager.kill_process(id2).unwrap();
        
        // Check that both are gone
        assert_eq!(manager.get_process_count(), 1); // kill removes, stop keeps
        assert_eq!(manager.get_running_count(), 0);
    }
}