use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::interval;
use tokio::signal;
use tracing::{info, error, warn, trace};
use num_cpus;

use crate::logging::{SwitchLogger, create_switch_event, create_failed_switch_event, calculate_decision_factors};
use crate::metrics::{MetricsCollector, AggregatedMetrics};
use crate::mcp_client::{McpClient, RunSchedulerRequest, StopSchedulerRequest};
use crate::policy::{PolicyEngine, SchedulerRecommendation, WorkloadType, PerformanceFeedback};

/// Helper function to get formatted timestamp
fn get_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("[{}]", now)
}

/// Helper function to clean up all schedcp temporary directories
async fn cleanup_schedcp_temp_directories() -> Result<()> {
    use tokio::process::Command;

    // Find and remove all .tmp directories that might be created by schedcp
    let output = Command::new("find")
        .args(&["/tmp", "-name", ".tmp*", "-type", "d"])
        .output()
        .await
        .context("Failed to find schedcp temporary directories")?;

    if output.status.success() {
        let temp_dirs = String::from_utf8_lossy(&output.stdout);
        for line in temp_dirs.lines() {
            let path = line.trim();
            if !path.is_empty() {
                let rm_output = Command::new("rm")
                    .arg("-rf")
                    .arg(path)
                    .output()
                    .await
                    .context("Failed to remove temporary directory")?;

                if !rm_output.status.success() {
                    eprintln!("Warning: Failed to remove temporary directory {}: {}",
                        path, String::from_utf8_lossy(&rm_output.stderr));
                } else {
                    println!("Cleaned up schedcp temporary directory: {}", path);
                }
            }
        }
    }

    Ok(())
}

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
    /// Path to the schedcp-cli binary
    pub schedcp_cli_path: String,
    /// Path to the switching events log file
    pub log_file_path: String,
    /// Log level for tracing (trace, debug, info, warn, error)
    pub log_level: String,
    /// System profile for adaptive tuning
    pub system_profile: SystemProfile,
    /// Enable performance feedback loop
    pub enable_performance_feedback: bool,
    /// Stabilization period after switch (in seconds)
    pub stabilization_period_secs: u64,
    /// Minimum performance improvement threshold (0.0-1.0)
    pub min_performance_improvement: f64,
}

/// System profile for different types of systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemProfile {
    /// Profile type (desktop, server, laptop, etc.)
    pub profile_type: ProfileType,
    /// Number of CPU cores
    pub cpu_cores: usize,
    /// Total memory in GB
    pub total_memory_gb: f64,
    /// Storage type (ssd, hdd, nvme)
    pub storage_type: StorageType,
    /// Primary use case
    pub primary_use_case: UseCase,
}

/// System profile types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProfileType {
    Desktop,
    Server,
    Laptop,
    Embedded,
    VirtualMachine,
}

/// Storage types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StorageType {
    SSD,
    HDD,
    NVMe,
    Hybrid,
}

/// Primary use cases
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UseCase {
    General,
    Gaming,
    Development,
    Database,
    WebServer,
    HighPerformanceComputing,
}

impl Default for SystemProfile {
    fn default() -> Self {
        Self {
            profile_type: ProfileType::Desktop,
            cpu_cores: num_cpus::get(),
            total_memory_gb: Self::detect_memory_gb(),
            storage_type: StorageType::SSD,
            primary_use_case: UseCase::General,
        }
    }
}

impl SystemProfile {
    fn detect_memory_gb() -> f64 {
        use procfs::Meminfo;
        if let Ok(meminfo) = Meminfo::new() {
            meminfo.mem_total as f64 / (1024.0 * 1024.0 * 1024.0) // Convert to GB
        } else {
            8.0 // Default fallback
        }
    }

    /// Get adaptive confidence threshold based on system profile
    pub fn get_adaptive_confidence_threshold(&self) -> f64 {
        match (self.profile_type, self.primary_use_case) {
            (ProfileType::Desktop, UseCase::Gaming) => 0.6, // More aggressive for gaming
            (ProfileType::Server, _) => 0.8,               // More conservative for servers
            (ProfileType::Laptop, _) => 0.7,               // Balanced for laptops
            _ => 0.7,                                      // Default
        }
    }

    /// Get adaptive switch interval based on system profile
    pub fn get_adaptive_switch_interval(&self) -> u64 {
        match (self.profile_type, self.primary_use_case) {
            (ProfileType::Desktop, UseCase::Gaming) => 5,   // Faster switching for gaming
            (ProfileType::Server, _) => 30,                 // Slower for server stability
            (ProfileType::Laptop, _) => 10,                 // Balanced for laptops
            _ => 10,                                        // Default
        }
    }

    /// Get adaptive aggregation window based on system profile
    pub fn get_adaptive_aggregation_window(&self) -> u64 {
        match self.profile_type {
            ProfileType::Desktop => 3,
            ProfileType::Server => 10,  // Longer window for stability
            ProfileType::Laptop => 5,
            _ => 5,
        }
    }
}

impl Default for DaemonConfig {
    fn default() -> Self {
        let system_profile = SystemProfile::default();
        Self {
            collection_interval_secs: 1,
            aggregation_window_secs: system_profile.get_adaptive_aggregation_window(),
            min_confidence_threshold: system_profile.get_adaptive_confidence_threshold(),
            min_switch_interval_secs: system_profile.get_adaptive_switch_interval(),
            enable_auto_switch: true,
            schedcp_cli_path: "/usr/local/bin/schedcp-cli".to_string(),
            log_file_path: "/var/log/autosa/switching_events.log".to_string(),
            log_level: "warn".to_string(),
            system_profile,
            enable_performance_feedback: true,
            stabilization_period_secs: 30,
            min_performance_improvement: 0.1,
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
    mcp_client: McpClient,
    current_execution_id: Arc<Mutex<Option<String>>>,
    switch_logger: SwitchLogger,
    performance_feedback: PerformanceFeedback,
}

impl AutoSchedulerDaemon {
    pub fn new(config: DaemonConfig) -> Result<Self> {
        let mcp_client = McpClient::new(config.schedcp_cli_path.clone());
        let switch_logger = SwitchLogger::new(config.log_file_path.clone());

        // Initialize logging system
        switch_logger.init_logging(&config.log_level)?;

        Ok(Self {
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
            mcp_client,
            current_execution_id: Arc::new(Mutex::new(None)),
            switch_logger,
            performance_feedback: PerformanceFeedback::new(),
        })
    }

    /// Start the daemon
    pub async fn start(&mut self) -> Result<()> {
        {
            let mut state = self.state.lock().await;
            state.is_running = true;
            let timestamp = get_timestamp();
            println!("{} Automatic scheduler adjustment daemon started", timestamp);
        }

        // Set up signal handling for graceful shutdown
        let state_clone = self.state.clone();
        let _switch_logger_clone = self.switch_logger.clone();
        let current_execution_id_clone = self.current_execution_id.clone();
        let mcp_client_clone = self.mcp_client.clone();

        tokio::spawn(async move {
            match signal::ctrl_c().await {
                Ok(()) => {
                    let timestamp = get_timestamp();
                    println!("\n{} Received Ctrl+C, shutting down gracefully...", timestamp);

                    // Stop the daemon
                    {
                        let mut state = state_clone.lock().await;
                        state.is_running = false;
                    }

                    // Stop current scheduler and restore default
                    {
                        let execution_id = current_execution_id_clone.lock().await;
                        if let Some(id) = &*execution_id {
                            use crate::mcp_client::StopSchedulerRequest;
                            let stop_request = StopSchedulerRequest {
                                execution_id: id.clone(),
                            };

                            if let Err(e) = mcp_client_clone.stop_scheduler(stop_request) {
                                eprintln!("{} Failed to stop scheduler: {}", timestamp, e);
                            } else {
                                println!("{} Scheduler stopped, disabling sched_ext", timestamp);
                            }
                        }
                    }

                    // Disable sched_ext to restore system default scheduler
                    println!("{} Disabling sched_ext to restore system default scheduler", timestamp);
                    if let Ok(output) = tokio::process::Command::new("sh")
                        .arg("-c")
                        .arg("echo 'bye' > /sys/kernel/sched_ext/state 2>/dev/null || echo 'Failed to disable sched_ext'")
                        .output()
                        .await
                    {
                        if output.status.success() {
                            println!("{} Successfully disabled sched_ext", timestamp);
                        } else {
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            eprintln!("{} Failed to disable sched_ext: {}", timestamp, stderr);
                        }
                    }

                    // Clean up temporary directories
                    if let Err(e) = cleanup_schedcp_temp_directories().await {
                        eprintln!("{} Failed to cleanup temporary directories: {}", timestamp, e);
                    }

                    std::process::exit(0);
                }
                Err(err) => {
                    eprintln!("Unable to listen for shutdown signal: {}", err);
                }
            }
        });

        // Start the monitoring loop
        self.monitoring_loop().await?;

        Ok(())
    }

    /// Stop the daemon
    pub async fn stop(&self) -> Result<()> {
        let mut state = self.state.lock().await;
        state.is_running = false;
        let timestamp = get_timestamp();
        println!("{} Automatic scheduler adjustment daemon stopped", timestamp);

        // Stop the current scheduler and restore default
        drop(state); // Release the lock before calling stop_current_scheduler
        self.stop_current_scheduler().await?;

        // Clean up any schedcp temporary directories that might be left behind
        cleanup_schedcp_temp_directories().await?;

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
    async fn analyze_and_adjust(&mut self) -> Result<()> {
        // Calculate aggregated metrics over our window
        let aggregated = self.metrics_collector.calculate_aggregated_metrics(
            self.config.aggregation_window_secs
        );

        if let Some(metrics) = aggregated {
            // Update performance feedback
            self.performance_feedback.update_metrics(&metrics);

            // Check if we need to rollback due to poor performance
            let current_scheduler = {
                let state = self.state.lock().await;
                state.current_scheduler.clone()
            };

            let current_workload = {
                let state = self.state.lock().await;
                state.current_workload_type.clone()
            };

            if let (Some(ref scheduler), Some(ref workload)) = (current_scheduler, current_workload) {
                if self.performance_feedback.should_rollback(scheduler, workload) {
                    let timestamp = get_timestamp();
                    println!("{} Performance degradation detected, considering rollback", timestamp);

                    // Try to get the best scheduler for this workload
                    if let Some(best_scheduler) = self.performance_feedback.get_best_scheduler(workload) {
                        if best_scheduler != *scheduler {
                            println!("{} Rolling back to better performing scheduler: {}", timestamp, best_scheduler);
                            // Create a rollback recommendation
                            let rollback_rec = SchedulerRecommendation {
                                scheduler_name: best_scheduler.clone(),
                                confidence: 0.9, // High confidence for rollback
                                reason: "Rollback due to performance degradation".to_string(),
                                suggested_args: self.policy_engine.get_scheduler_info(&best_scheduler)
                                    .map(|info| info.default_args.clone())
                                    .unwrap_or_default(),
                            };
                            self.switch_scheduler(&rollback_rec, &metrics, workload).await?;
                            return Ok(());
                        }
                    }
                }
            }

            // Classify workload (need mutable reference)
            let workload_type = self.policy_engine.classify_workload(&metrics);

            // Get scheduler recommendation
            let recommendation = self.policy_engine.recommend_scheduler(&metrics);

            // Check if there's a significant change to report
            let should_report = {
                let mut state = self.state.lock().await;

                // Check if workload type changed
                let workload_changed = state.current_workload_type.as_ref() != Some(&workload_type);

                // Check if recommendation changed significantly (different scheduler or confidence change > 10%)
                let recommendation_changed = match &state.last_recommendation {
                    Some(last_rec) => {
                        last_rec.scheduler_name != recommendation.scheduler_name ||
                        (last_rec.confidence - recommendation.confidence).abs() > 0.1
                    },
                    None => true,
                };

                // Update state
                state.current_workload_type = Some(workload_type.clone());
                state.last_recommendation = Some(recommendation.clone());

                workload_changed || recommendation_changed
            };

            // Only print significant changes
            if should_report {
                let timestamp = get_timestamp();
                println!("{} Workload classified as: {:?}", timestamp, workload_type);
                println!("{} Recommended scheduler: {} (confidence: {:.2}%) - {}",
                    timestamp,
                    recommendation.scheduler_name,
                    recommendation.confidence * 100.0,
                    recommendation.reason
                );
            }

            // Check if we should switch schedulers
            if self.should_switch_scheduler(&recommendation, &metrics, &workload_type).await? {
                self.switch_scheduler(&recommendation, &metrics, &workload_type).await?;
            }
        }

        Ok(())
    }

    /// Determine if we should switch to a new scheduler
    async fn should_switch_scheduler(&self, recommendation: &SchedulerRecommendation, metrics: &AggregatedMetrics, workload_type: &WorkloadType) -> Result<bool> {
        // Check if auto-switching is enabled
        if !self.config.enable_auto_switch {
            let reason = "Auto-switching is disabled";
            trace!("Not switching scheduler: {}", reason);

            let current_scheduler = {
                let state = self.state.lock().await;
                state.current_scheduler.clone()
            };

            // Only log no-switch decisions at trace level to avoid log spam
            if tracing::level_enabled!(tracing::Level::TRACE) {
                if let Some(ref scheduler) = current_scheduler {
                    self.switch_logger.log_no_switch_decision(
                        Some(scheduler),
                        recommendation,
                        metrics,
                        workload_type,
                        reason,
                    )?;
                }
            }
            return Ok(false);
        }

        // Use adaptive confidence threshold based on system profile
        let adaptive_threshold = self.config.system_profile.get_adaptive_confidence_threshold();

        // Check confidence threshold
        if recommendation.confidence < adaptive_threshold {
            let reason = format!("Confidence {:.2} below adaptive threshold {:.2}",
                recommendation.confidence, adaptive_threshold);
            trace!("Not switching scheduler: {}", reason);

            let current_scheduler = {
                let state = self.state.lock().await;
                state.current_scheduler.clone()
            };

            // Only log no-switch decisions at trace level to avoid log spam
            if tracing::level_enabled!(tracing::Level::TRACE) {
                if let Some(ref scheduler) = current_scheduler {
                    self.switch_logger.log_no_switch_decision(
                        Some(scheduler),
                        recommendation,
                        metrics,
                        workload_type,
                        &reason,
                    )?;
                }
            }
            return Ok(false);
        }

        // Check if workload is stable enough for a switch
        if !self.policy_engine.is_workload_stable() {
            let reason = "Workload not stable enough for switching";
            trace!("Not switching scheduler: {}", reason);

            let current_scheduler = {
                let state = self.state.lock().await;
                state.current_scheduler.clone()
            };

            if tracing::level_enabled!(tracing::Level::TRACE) {
                if let Some(ref scheduler) = current_scheduler {
                    self.switch_logger.log_no_switch_decision(
                        Some(scheduler),
                        recommendation,
                        metrics,
                        workload_type,
                        reason,
                    )?;
                }
            }
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
            None => {
                info!("No current scheduler, will switch to {}", recommendation.scheduler_name);
                return Ok(true);
            }
        };

        // Don't switch if it's the same scheduler
        if current_scheduler == recommendation.scheduler_name {
            let reason = "Same scheduler already running";
            trace!("Not switching scheduler: {}", reason);

            // Only log no-switch decisions at trace level to avoid log spam
            if tracing::level_enabled!(tracing::Level::TRACE) {
                self.switch_logger.log_no_switch_decision(
                    Some(&current_scheduler),
                    recommendation,
                    metrics,
                    workload_type,
                    &reason,
                )?;
            }
            return Ok(false);
        }

        // Dynamic switch interval based on workload volatility
        let dynamic_min_interval = self.calculate_dynamic_switch_interval(workload_type, metrics);

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
            if now - last_switch < dynamic_min_interval {
                let reason = format!("Dynamic minimum switch interval not reached ({} < {} seconds)",
                    now - last_switch, dynamic_min_interval);
                trace!("Not switching scheduler: {}", reason);

                self.switch_logger.log_no_switch_decision(
                    Some(&current_scheduler),
                    recommendation,
                    metrics,
                    workload_type,
                    &reason,
                )?;
                return Ok(false);
            }
        }

        // Check if the potential performance gain justifies the switch cost
        if self.is_switch_cost_justified(&current_scheduler, recommendation, workload_type)? {
            Ok(true)
        } else {
            let reason = "Performance gain does not justify switch cost";
            trace!("Not switching scheduler: {}", reason);

            self.switch_logger.log_no_switch_decision(
                Some(&current_scheduler),
                recommendation,
                metrics,
                workload_type,
                reason,
            )?;
            Ok(false)
        }
    }

    /// Calculate dynamic switch interval based on workload characteristics and system profile
    fn calculate_dynamic_switch_interval(&self, workload_type: &WorkloadType, metrics: &AggregatedMetrics) -> u64 {
        let base_interval = self.config.system_profile.get_adaptive_switch_interval();

        // Adjust interval based on workload type
        let workload_multiplier = match workload_type {
            WorkloadType::LatencySensitive => 2.0, // Longer intervals for latency-sensitive
            WorkloadType::Transitioning => 0.5,    // Shorter for transitioning workloads
            WorkloadType::Batch => 3.0,             // Much longer for stable batch workloads
            _ => 1.0,
        };

        // Adjust based on volatility (CPU variance)
        let volatility_factor = if metrics.cpu_max_percent > metrics.cpu_avg_percent {
            let variance = metrics.cpu_max_percent - metrics.cpu_avg_percent;
            if variance > 30.0 {
                0.5 // High volatility - shorter intervals
            } else if variance > 15.0 {
                0.8 // Medium volatility
            } else {
                1.2 // Low volatility - longer intervals
            }
        } else {
            1.0
        };

        // Adjust based on system profile
        let profile_factor = match self.config.system_profile.profile_type {
            ProfileType::Desktop => 1.0,
            ProfileType::Server => 2.0,     // Slower switching for servers
            ProfileType::Laptop => 1.5,     // Moderate for laptops
            ProfileType::Embedded => 0.8,   // Faster for embedded systems
            ProfileType::VirtualMachine => 0.7, // Faster for VMs
        };

        let dynamic_interval = (base_interval as f64 * workload_multiplier * volatility_factor * profile_factor) as u64;

        // Clamp to reasonable bounds (5 seconds to 5 minutes)
        dynamic_interval.clamp(5, 300)
    }

    /// Check if the performance gain justifies the cost of switching
    fn is_switch_cost_justified(&self, current_scheduler: &str, recommendation: &SchedulerRecommendation, workload_type: &WorkloadType) -> Result<bool> {
        // Get historical performance data
        let current_key = format!("{}-{:?}", current_scheduler, workload_type);
        let recommended_key = format!("{}-{:?}", recommendation.scheduler_name, workload_type);

        let current_score = self.performance_feedback.performance_history
            .get(&current_key)
            .map(|p| p.get_overall_score())
            .unwrap_or(0.5); // Default score if no history

        let recommended_score = self.performance_feedback.performance_history
            .get(&recommended_key)
            .map(|p| p.get_overall_score())
            .unwrap_or(0.5);

        // Use adaptive improvement threshold based on system profile
        let improvement_threshold = match self.config.system_profile.primary_use_case {
            UseCase::Gaming => 0.05,      // More aggressive for gaming
            UseCase::HighPerformanceComputing => 0.15, // More conservative for HPC
            _ => self.config.min_performance_improvement,
        };

        let relative_improvement = (recommended_score - current_score) / f64::max(current_score, 0.1);

        Ok(relative_improvement > improvement_threshold)
    }

    /// Switch to a new scheduler using the MCP client
    async fn switch_scheduler(&mut self, recommendation: &SchedulerRecommendation, metrics: &AggregatedMetrics, workload_type: &WorkloadType) -> Result<()> {
        let start_time = SystemTime::now();
        let now = start_time
            .duration_since(UNIX_EPOCH)
            .context("Failed to get current time")?
            .as_secs();

        info!("Switching to scheduler: {} with args: {:?}",
            recommendation.scheduler_name,
            recommendation.suggested_args
        );

        // Get current scheduler for logging
        let previous_scheduler = {
            let state = self.state.lock().await;
            state.current_scheduler.clone()
        };

        // Calculate decision factors
        let decision_factors = calculate_decision_factors(
            previous_scheduler.as_deref(),
            recommendation,
            self.config.enable_auto_switch,
            self.config.min_confidence_threshold,
            true, // We're here because min interval was met
        );

        // Record the switch for performance tracking
        self.performance_feedback.record_switch(&recommendation.scheduler_name, workload_type);

        // Stop the current scheduler if there is one running
        {
            let execution_id = self.current_execution_id.lock().await;
            if let Some(id) = &*execution_id {
                info!("Stopping current scheduler with execution ID: {}", id);
                let stop_request = StopSchedulerRequest {
                    execution_id: id.clone(),
                };

                match self.mcp_client.stop_scheduler(stop_request) {
                    Ok(response) => {
                        info!("Successfully stopped scheduler: {}", response.message);
                    }
                    Err(e) => {
                        warn!("Failed to stop current scheduler: {}", e);
                        // Continue anyway, as we might still want to start the new scheduler
                    }
                }
            }
        }

        // Clean up any temporary directories left by the previous scheduler
        cleanup_schedcp_temp_directories().await?;

        // Start the new scheduler
        let run_request = RunSchedulerRequest {
            name: recommendation.scheduler_name.clone(),
            args: recommendation.suggested_args.clone(),
        };

        match self.mcp_client.run_scheduler(run_request) {
            Ok(response) => {
                let switch_duration = start_time.elapsed()
                    .context("Failed to calculate switch duration")?
                    .as_millis() as u64;

                info!("Successfully started scheduler: {}", response.message);

                // Update our state
                {
                    let mut state = self.state.lock().await;
                    state.current_scheduler = Some(recommendation.scheduler_name.clone());
                    state.last_switch_timestamp = Some(now);
                }

                // Store the new execution ID
                {
                    let mut execution_id = self.current_execution_id.lock().await;
                    *execution_id = Some(response.execution_id.clone());
                }

                info!("Scheduler switched to {} at {}", recommendation.scheduler_name, now);

                // Log the successful switch
                let switch_event = create_switch_event(
                    previous_scheduler,
                    recommendation,
                    metrics,
                    workload_type,
                    decision_factors,
                    Some(response.execution_id),
                    Some(switch_duration),
                );

                if let Err(e) = self.switch_logger.log_switch_event(&switch_event) {
                    error!("Failed to log switching event: {}", e);
                }
            }
            Err(e) => {
                error!("Failed to start new scheduler: {}", e);

                // Log the failed switch
                let switch_event = create_failed_switch_event(
                    previous_scheduler,
                    recommendation,
                    metrics,
                    workload_type,
                    decision_factors,
                    e.to_string(),
                );

                if let Err(log_err) = self.switch_logger.log_switch_event(&switch_event) {
                    error!("Failed to log failed switching event: {}", log_err);
                }

                return Err(e);
            }
        }

        Ok(())
    }

    /// Get current daemon state
    pub async fn get_state(&self) -> DaemonState {
        self.state.lock().await.clone()
    }

    /// Stop the current scheduler and restore system default
    async fn stop_current_scheduler(&self) -> Result<()> {
        let timestamp = get_timestamp();

        // Stop the current scheduler if there is one running
        {
            let execution_id = self.current_execution_id.lock().await;
            if let Some(id) = &*execution_id {
                println!("{} Stopping current scheduler with execution ID: {}", timestamp, id);
                let stop_request = StopSchedulerRequest {
                    execution_id: id.clone(),
                };

                match self.mcp_client.stop_scheduler(stop_request) {
                    Ok(response) => {
                        println!("{} Successfully stopped scheduler: {}", timestamp, response.message);
                    }
                    Err(e) => {
                        eprintln!("{} Failed to stop current scheduler: {}", timestamp, e);
                    }
                }
            }
        }

        // Disable sched_ext to restore system default scheduler
        println!("{} Disabling sched_ext to restore system default scheduler", timestamp);
        match tokio::process::Command::new("sh")
            .arg("-c")
            .arg("echo 'bye' > /sys/kernel/sched_ext/state 2>/dev/null || echo 'Failed to disable sched_ext'")
            .output()
            .await
        {
            Ok(output) => {
                if output.status.success() {
                    println!("{} Successfully disabled sched_ext", timestamp);
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    eprintln!("{} Failed to disable sched_ext: {}", timestamp, stderr);
                }
            }
            Err(e) => {
                eprintln!("{} Error disabling sched_ext: {}", timestamp, e);
            }
        }

        // Clear the execution ID
        {
            let mut execution_id = self.current_execution_id.lock().await;
            *execution_id = None;
        }

        // Update state
        {
            let mut state = self.state.lock().await;
            state.current_scheduler = None;
        }

        Ok(())
    }

    /// Get configuration
    pub fn get_config(&self) -> &DaemonConfig {
        &self.config
    }

    /// Update configuration
    pub async fn update_config(&mut self, new_config: DaemonConfig) {
        self.config = new_config;
    }

    /// Update system profile based on current system detection
    pub fn detect_system_profile(&mut self) {
        self.config.system_profile = SystemProfile::default();

        // Auto-detect profile type based on system characteristics
        self.config.system_profile.profile_type = if self.config.system_profile.cpu_cores >= 16 {
            ProfileType::Server
        } else if self.config.system_profile.total_memory_gb < 8.0 {
            ProfileType::Laptop
        } else {
            ProfileType::Desktop
        };

        // Auto-detect storage type (simplified)
        self.config.system_profile.storage_type = StorageType::SSD; // Default assumption

        // Update adaptive parameters based on detected profile
        self.config.aggregation_window_secs = self.config.system_profile.get_adaptive_aggregation_window();
        self.config.min_confidence_threshold = self.config.system_profile.get_adaptive_confidence_threshold();
        self.config.min_switch_interval_secs = self.config.system_profile.get_adaptive_switch_interval();
    }
}
