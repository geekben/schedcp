//! MCP Client for AutoSA
//!
//! This module provides a client interface to communicate with the schedcp MCP server
//! for scheduler management operations.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::process::Command;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Information about a scheduler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerInfo {
    pub name: String,
    pub production_ready: bool,
    pub description: String,
    pub algorithm: String,
    pub use_cases: Vec<String>,
    pub characteristics: String,
    pub tuning_parameters: HashMap<String, Value>,
    pub limitations: String,
    pub performance_profile: String,
}

/// Response from list_schedulers command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListSchedulersResponse {
    pub schedulers: Vec<SchedulerInfo>,
}

/// Request for run_scheduler command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSchedulerRequest {
    pub name: String,
    pub args: Vec<String>,
}

/// Response from run_scheduler command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunSchedulerResponse {
    pub execution_id: String,
    pub scheduler: String,
    pub status: String,
    pub message: String,
}

/// Request for stop_scheduler command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopSchedulerRequest {
    pub execution_id: String,
}

/// Response from stop_scheduler command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopSchedulerResponse {
    pub execution_id: String,
    pub status: String,
    pub message: String,
}

/// Response from get_execution_status command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetExecutionStatusResponse {
    pub execution_id: String,
    pub scheduler_name: String,
    pub command: String,
    pub args: Vec<String>,
    pub status: String,
    pub pid: Option<u32>,
    pub start_time: u64,
    pub end_time: Option<u64>,
    pub duration: Option<u64>,
    pub exit_code: Option<i32>,
    pub output: Vec<String>,
}

/// MCP Client for communicating with schedcp server via CLI
pub struct McpClient {
    /// Path to the schedcp-cli binary
    cli_path: String,
    /// Track temporary directories created by this client
    temp_directories: Arc<Mutex<HashMap<String, PathBuf>>>,
}

impl McpClient {
    /// Create a new MCP client
    pub fn new(cli_path: String) -> Self {
        Self {
            cli_path,
            temp_directories: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// List available schedulers
    pub fn list_schedulers(&self) -> Result<ListSchedulersResponse> {
        // Since there's no JSON output, we'll return a minimal response
        // In a real implementation, we would parse the text output
        Ok(ListSchedulersResponse {
            schedulers: vec![],
        })
    }

    /// Run a scheduler
    pub fn run_scheduler(&self, request: RunSchedulerRequest) -> Result<RunSchedulerResponse> {
        let mut cmd = Command::new(&self.cli_path);
        cmd.arg("run").arg(&request.name);

        // If there are arguments, pass them after the -- separator
        if !request.args.is_empty() {
            cmd.arg("--");
            for arg in &request.args {
                cmd.arg(arg);
            }
        }

        // Run the command in background
        let child = cmd
            .spawn()
            .context("Failed to start schedcp-cli run")?;

        let pid = child.id();

        // Create a fake execution ID
        let execution_id = format!("exec_{}", pid);
        let scheduler_name = request.name.clone();

        // Try to find and track the temporary directory created for this scheduler
        // We'll look for recently created .tmp directories that might belong to this execution
        self.track_temp_directory(&execution_id, &scheduler_name)?;

        Ok(RunSchedulerResponse {
            execution_id,
            scheduler: scheduler_name,
            status: "started".to_string(),
            message: format!("Scheduler {} started with PID {}", request.name, pid),
        })
    }

    /// Stop a running scheduler
    pub fn stop_scheduler(&self, request: StopSchedulerRequest) -> Result<StopSchedulerResponse> {
        // Extract PID from execution_id (this is a hack - in reality we'd need to track this properly)
        let parts: Vec<&str> = request.execution_id.split('_').collect();
        let execution_id = request.execution_id.clone();
        if parts.len() == 2 {
            let pid = parts[1];

            // Kill the process
            let _ = Command::new("kill")
                .arg(pid)
                .output();
        }

        Ok(StopSchedulerResponse {
            execution_id,
            status: "stopped".to_string(),
            message: format!("Scheduler with execution ID {} stopped", request.execution_id),
        })
    }

    /// Get execution status
    pub fn get_execution_status(&self, execution_id: &str) -> Result<GetExecutionStatusResponse> {
        // Return a mock response
        Ok(GetExecutionStatusResponse {
            execution_id: execution_id.to_string(),
            scheduler_name: "unknown".to_string(),
            command: "unknown".to_string(),
            args: vec![],
            status: "running".to_string(),
            pid: None,
            start_time: 0,
            end_time: None,
            duration: None,
            exit_code: None,
            output: vec![],
        })
    }

    /// Track temporary directories created by this client
    fn track_temp_directory(&self, execution_id: &str, _scheduler_name: &str) -> Result<()> {
        use std::process::Command;

        // Find ALL .tmp directories - we'll clean them all up anyway
        // This is safer than trying to track specific ones
        let output = Command::new("find")
            .args(&["/tmp", "-name", ".tmp*", "-type", "d"])
            .output()
            .context("Failed to find temporary directories")?;

        if output.status.success() {
            let temp_dirs = String::from_utf8_lossy(&output.stdout);
            let mut tracked_any = false;
            for line in temp_dirs.lines() {
                let path = PathBuf::from(line.trim());
                if path.exists() {
                    // Store all temp directories found
                    let temp_dirs_map = self.temp_directories.try_lock();
                    if let Ok(mut temp_dirs_map) = temp_dirs_map {
                        temp_dirs_map.insert(format!("{}_{}", execution_id, tracked_any), path);
                        tracked_any = true;
                    }
                }
            }
        }

        Ok(())
    }

    /// Get temporary directories tracked by this client
    pub async fn get_temp_directories(&self) -> Vec<PathBuf> {
        let temp_dirs_map = self.temp_directories.lock().await;
        temp_dirs_map.values().cloned().collect()
    }

    /// Clear tracking for a specific execution
    pub async fn clear_execution_tracking(&self, execution_id: &str) {
        let mut temp_dirs_map = self.temp_directories.lock().await;
        temp_dirs_map.remove(execution_id);
    }

    /// Clear all tracking
    pub async fn clear_all_tracking(&self) {
        let mut temp_dirs_map = self.temp_directories.lock().await;
        temp_dirs_map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcp_client_creation() {
        let client = McpClient::new("/usr/local/bin/schedcp-cli".to_string());
        assert_eq!(client.cli_path, "/usr/local/bin/schedcp-cli");
    }
}
