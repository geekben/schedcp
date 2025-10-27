use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::interval;

use crate::metrics::MetricsCollector;
use crate::policy::{PolicyEngine, SchedulerRecommendation, WorkloadType};

/// Configuration for the automatic scheduler adjustment daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonConfig {
    /// How often to collect metrics (in seconds)
    pub collection_interval_secs: u64,
    /// Time window for calculating aggregated metrics (in seconds)
    pub aggregation_window_secs: u64,
    /// Minimum confidence level required to switch schedulers
    pub min_confidence_threshold: f64,
    /// Minimum time between scheduler switches (in seconds)
    pub min_switch_interval_secs: u64,
    /// Enable automatic scheduler switching
    pub enable_auto_switch: bool,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            collection_interval_secs: 1,
            aggregation_window_secs: 60,
            min_confidence_threshold: 0.7,
            min_switch_interval_secs: 300, // 5 minutes
            enable_auto_switch: true,
        }
    }
}

/// Current state of the daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonState {
    pub is_running: bool,
    pub current_scheduler: Option<String>,
    pub last_switch_timestamp: Option<u64>,
    pub current_workload_type: Option<WorkloadType>,
    pub last_recommendation: Option<SchedulerRecommendation>,
    pub metrics_samples_collected: usize,
}

/// Automatic scheduler adjustment daemon
pub struct AutoSchedulerDaemon {
    config: DaemonConfig,
    policy_engine: PolicyEngine,
    metrics_collector: MetricsCollector,
    state: Arc<Mutex<DaemonState>>,
    // We would integrate with the MCP server here for actual scheduler management
    // For now, we'll simulate the interface
}

impl AutoSchedulerDaemon {
    pub fn new(config: DaemonConfig) -> Self {
        Self {
            config,
            policy_engine: PolicyEngine::new(),
            metrics_collector: MetricsCollector::new(3600), // Keep 1 hour of samples at 1/sec
            state: Arc::new(Mutex::new(DaemonState {
                is_running: false,
                current_scheduler: None,
                last_switch_timestamp: None,
                current_workload_type: None,
                last_recommendation: None,
                metrics_samples_collected: 0,
            })),
        }
    }

    /// Start the daemon
    pub async fn start(&mut self) -> Result<()> {
        {
            let mut state = self.state.lock().await;
            state.is_running = true;
            println!("Automatic scheduler adjustment daemon started");
        }

        // Start the monitoring loop
        self.monitoring_loop().await?;

        Ok(())
    }

    /// Stop the daemon
    pub async fn stop(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        state.is_running = false;
        println!("Automatic scheduler adjustment daemon stopped");
        Ok(())
    }

    /// Main monitoring loop
    async fn monitoring_loop(&mut self) -> Result<()> {
        let mut interval = interval(Duration::from_secs(self.config.collection_interval_secs));

        loop {
            interval.tick().await;

            // Check if we should still be running
            {
                let state = self.state.lock().await;
                if !state.is_running {
                    break;
                }
            }

            // Collect metrics
            match self.metrics_collector.collect_metrics().await {
                Ok(metrics) => {
                    self.metrics_collector.add_sample(metrics);

                    // Update sample count
                    {
                        let mut state = self.state.lock().await;
                        state.metrics_samples_collected += 1;
                    }

                    // Analyze metrics and potentially switch schedulers
                    self.analyze_and_adjust().await?;
                }
                Err(e) => {
                    eprintln!("Failed to collect metrics: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Analyze collected metrics and adjust scheduler if needed
    async fn analyze_and_adjust(&self) -> Result<()> {
        // Calculate aggregated metrics over our window
        let aggregated = self.metrics_collector.calculate_aggregated_metrics(
            self.config.aggregation_window_secs
        );

        if let Some(metrics) = aggregated {
            // Classify workload
            let workload_type = self.policy_engine.classify_workload(&metrics);

            // Update state with current workload type
            {
                let mut state = self.state.lock().await;
                state.current_workload_type = Some(workload_type.clone());
            }

            // Get scheduler recommendation
            let recommendation = self.policy_engine.recommend_scheduler(&metrics);

            // Update state with recommendation
            {
                let mut state = self.state.lock().await;
                state.last_recommendation = Some(recommendation.clone());
            }

            println!("Workload classified as: {:?}", workload_type);
            println!("Recommended scheduler: {} (confidence: {:.2}%) - {}",
                recommendation.scheduler_name,
                recommendation.confidence * 100.0,
                recommendation.reason
            );

            // Check if we should switch schedulers
            if self.should_switch_scheduler(&recommendation).await? {
                self.switch_scheduler(&recommendation).await?;
            }
        }

        Ok(())
    }

    /// Determine if we should switch to a new scheduler
    async fn should_switch_scheduler(&self, recommendation: &SchedulerRecommendation) -> Result<bool> {
        // Check if auto-switching is enabled
        if !self.config.enable_auto_switch {
            return Ok(false);
        }

        // Check confidence threshold
        if recommendation.confidence < self.config.min_confidence_threshold {
            return Ok(false);
        }

        // Check if we have a current scheduler
        let current_scheduler = {
            let state = self.state.lock().await;
            state.current_scheduler.clone()
        };

        // If no current scheduler, we should switch
        let current_scheduler = match current_scheduler {
            Some(scheduler) => scheduler,
            None => return Ok(true),
        };

        // Don't switch if it's the same scheduler
        if current_scheduler == recommendation.scheduler_name {
            return Ok(false);
        }

        // Check minimum switch interval
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Failed to get current time")?
            .as_secs();

        let last_switch_time = {
            let state = self.state.lock().await;
            state.last_switch_timestamp
        };

        if let Some(last_switch) = last_switch_time {
            if now - last_switch < self.config.min_switch_interval_secs {
                println!("Not switching scheduler - minimum interval not reached");
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Switch to a new scheduler (simulated)
    async fn switch_scheduler(&self, recommendation: &SchedulerRecommendation) -> Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Failed to get current time")?
            .as_secs();

        println!("Switching to scheduler: {} with args: {:?}",
            recommendation.scheduler_name,
            recommendation.suggested_args
        );

        // In a real implementation, we would:
        // 1. Stop the current scheduler using the MCP server
        // 2. Start the new scheduler using the MCP server
        // 3. Update our state

        // For now, just update our state
        {
            let mut state = self.state.lock().await;
            state.current_scheduler = Some(recommendation.scheduler_name.clone());
            state.last_switch_timestamp = Some(now);
        }

        // Log the switch
        println!("Scheduler switched to {} at {}", recommendation.scheduler_name, now);

        Ok(())
    }

    /// Get current daemon state
    pub async fn get_state(&self) -> DaemonState {
        self.state.lock().await.clone()
    }

    /// Get configuration
    pub fn get_config(&self) -> &DaemonConfig {
        &self.config
    }

    /// Update configuration
    pub async fn update_config(&mut self, new_config: DaemonConfig) {
        self.config = new_config;
    }
}