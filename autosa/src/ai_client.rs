use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::sync::Arc;
use tokio::{process, sync::Mutex, time::Instant};
use tracing::{info, warn, error, debug};

use crate::metrics::AggregatedMetrics;
use crate::policy::{WorkloadType, SchedulerRecommendation};

/// AI client for intelligent scheduler selection
pub struct AiClient {
    iflow_cli_path: String,
    model_name: String,
    enabled: bool,
    timeout_secs: u64,
    token_usage: Arc<Mutex<TokenUsage>>,
}

/// Token usage tracking
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub total_tokens: u64,
    pub prompt_tokens: u64,
    pub response_tokens: u64,
    pub total_requests: u64,
    pub total_response_time_ms: u64,
}

/// AI scheduler recommendation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSchedulerRecommendation {
    pub scheduler_name: String,
    pub confidence: f64,
    pub reasoning: String,
    pub suggested_args: Vec<String>,
    pub expected_benefit: Option<String>,
}

impl AiClient {
    pub fn new(iflow_cli_path: String, model_name: String) -> Self {
        Self {
            iflow_cli_path,
            model_name,
            enabled: true,
            timeout_secs: 30,
            token_usage: Arc::new(Mutex::new(TokenUsage::default())),
        }
    }

    /// Check if iFlow CLI is available and working
    pub async fn check_availability(&self) -> bool {
        if !self.enabled {
            return false;
        }

        // Try a simple test query
        let output = process::Command::new(&self.iflow_cli_path)
            .arg("-p")
            .arg("Respond with just: OK")
            .output()
            .await;

        match output {
            Ok(result) => {
                let stdout = String::from_utf8_lossy(&result.stdout);
                result.status.success() && stdout.contains("OK")
            },
            Err(e) => {
                warn!("iFlow CLI not available: {}", e);
                false
            }
        }
    }

    /// Get AI-based scheduler recommendation
    pub async fn recommend_scheduler(
        &self,
        metrics: &AggregatedMetrics,
        current_scheduler: Option<&str>,
        workload_type: &WorkloadType,
        historical_performance: &str,
    ) -> Result<Option<AiSchedulerRecommendation>> {
        if !self.enabled {
            return Ok(None);
        }

        let prompt = self.build_prompt(metrics, current_scheduler, workload_type, historical_performance);
        
        debug!("Sending prompt to AI: {}", prompt);
        
        // Track timing
        let start_time = Instant::now();
        let prompt_tokens = self.estimate_tokens(&prompt) as u64;

        let output = process::Command::new(&self.iflow_cli_path)
            .arg("-p")
            .arg(&prompt)
            .output()
            .await
            .context("Failed to execute iFlow CLI")?;

        let response_time = start_time.elapsed().as_millis() as u64;
        let response_str = String::from_utf8_lossy(&output.stdout);
        let response_tokens = self.estimate_tokens(&response_str) as u64;

        // Update token usage tracking
        {
            let mut usage = self.token_usage.lock().await;
            usage.total_tokens += prompt_tokens + response_tokens;
            usage.prompt_tokens += prompt_tokens;
            usage.response_tokens += response_tokens;
            usage.total_requests += 1;
            usage.total_response_time_ms += response_time;
            
            info!("Token usage - Request {}: {} prompt + {} response = {} total, {}ms", 
                usage.total_requests, prompt_tokens, response_tokens, 
                usage.total_tokens, response_time);
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("iFlow CLI error: {}", stderr);
            return Ok(None);
        }
        
        // Try to extract JSON from the response
        let json_str = if response_str.contains('{') && response_str.contains('}') {
            // Extract JSON block if present
            if let Some(start) = response_str.find('{') {
                if let Some(end) = response_str.rfind('}') {
                    &response_str[start..=end]
                } else {
                    &response_str[start..]
                }
            } else {
                &response_str
            }
        } else {
            // No JSON found, return None
            warn!("AI response doesn't contain JSON: {}", response_str);
            return Ok(None);
        };

        // Parse the JSON response
        match serde_json::from_str::<AiSchedulerRecommendation>(json_str) {
            Ok(recommendation) => {
                info!("AI recommended scheduler: {} with confidence: {:.2}", 
                    recommendation.scheduler_name, recommendation.confidence);
                Ok(Some(recommendation))
            }
            Err(e) => {
                error!("Failed to parse AI response: {}", e);
                error!("Extracted JSON: {}", json_str);
                error!("Raw response: {}", response_str);
                Ok(None)
            }
        }
    }

    /// Estimate token count (rough approximation: 1 token ≈ 4 characters)
    fn estimate_tokens(&self, text: &str) -> usize {
        (text.len() + 3) / 4
    }

    /// Get current token usage statistics
    pub async fn get_token_usage(&self) -> TokenUsage {
        self.token_usage.lock().await.clone()
    }

    /// Reset token usage statistics
    pub async fn reset_token_usage(&self) {
        let mut usage = self.token_usage.lock().await;
        *usage = TokenUsage::default();
    }

    /// Build compact prompt for AI scheduler selection
    fn build_prompt(
        &self,
        metrics: &AggregatedMetrics,
        current_scheduler: Option<&str>,
        workload_type: &WorkloadType,
        _historical_performance: &str,
    ) -> String {
        format!(
            r#"Select best sched_ext scheduler. Current: {}, Workload: {:?}. 
Metrics: CPU {:.1}%/{:.1}%, I/O wait {:.1}%, Mem {:.1}%/{:.1}%, I/O {:.1}/{:.1} B/s, queue {:.1}, timeslices {:.1}/s.

SCHEDULERS:
- bpfland: interactive/low latency
- flash: predictable/low latency  
- lavd: gaming/interactive
- rusty: balanced/CPU intensive
- simple: uniform workloads

Respond with JSON only: {{"scheduler_name": "...", "confidence": 0.95, "reasoning": "...", "suggested_args": []}}"#,
            current_scheduler.unwrap_or("none"),
            workload_type,
            metrics.cpu_avg_percent,
            metrics.cpu_max_percent,
            metrics.cpu_iowait_percent,
            metrics.memory_avg_percent,
            metrics.memory_max_percent,
            metrics.io_read_bytes_per_sec,
            metrics.io_write_bytes_per_sec,
            metrics.io_avg_queue_depth,
            metrics.sched_timeslices_per_sec
        )
    }

    /// Enable/disable AI client
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set timeout for AI requests
    pub fn set_timeout(&mut self, timeout_secs: u64) {
        self.timeout_secs = timeout_secs;
    }

    /// Get current status
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}