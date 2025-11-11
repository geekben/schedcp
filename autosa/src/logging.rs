use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::Write;
use tracing::{info, error};

use crate::metrics::AggregatedMetrics;
use crate::policy::{SchedulerRecommendation, WorkloadType};

/// Detailed switching event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwitchingEvent {
    pub timestamp: DateTime<Utc>,
    pub previous_scheduler: Option<String>,
    pub new_scheduler: String,
    pub new_scheduler_args: Vec<String>,
    pub workload_type: WorkloadType,
    pub confidence: f64,
    pub reason: String,
    pub metrics: AggregatedMetrics,
    pub decision_factors: DecisionFactors,
    pub execution_id: Option<String>,
    pub switch_duration_ms: Option<u64>,
    pub success: bool,
    pub error_message: Option<String>,
}

/// Factors that influenced the switching decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionFactors {
    pub cpu_threshold_met: bool,
    pub io_threshold_met: bool,
    pub memory_threshold_met: bool,
    pub confidence_threshold_met: bool,
    pub min_switch_interval_met: bool,
    pub auto_switch_enabled: bool,
    pub different_scheduler: bool,
}

/// Logger for switching events
#[derive(Clone)]
pub struct SwitchLogger {
    log_file_path: String,
}

impl SwitchLogger {
    pub fn new(log_file_path: String) -> Self {
        Self { log_file_path }
    }

    /// Log a switching event to file
    pub fn log_switch_event(&self, event: &SwitchingEvent) -> Result<()> {
        let json_line = serde_json::to_string(event)?;
        
        match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_file_path)
        {
            Ok(mut file) => {
                if let Err(e) = writeln!(file, "{}", json_line) {
                    error!("Failed to write to log file {}: {}", self.log_file_path, e);
                    return Err(e.into());
                }
                info!("Switching event logged to {}", self.log_file_path);
            }
            Err(e) => {
                error!("Failed to open log file {}: {}", self.log_file_path, e);
                return Err(e.into());
            }
        }

        Ok(())
    }

    /// Log a decision not to switch
    pub fn log_no_switch_decision(
        &self,
        current_scheduler: Option<&str>,
        recommendation: &SchedulerRecommendation,
        metrics: &AggregatedMetrics,
        workload_type: &WorkloadType,
        reason: &str,
    ) -> Result<()> {
        let event = SwitchingEvent {
            timestamp: Utc::now(),
            previous_scheduler: current_scheduler.map(|s| s.to_string()),
            new_scheduler: recommendation.scheduler_name.clone(),
            new_scheduler_args: recommendation.suggested_args.clone(),
            workload_type: workload_type.clone(),
            confidence: recommendation.confidence,
            reason: format!("NO_SWITCH: {}", reason),
            metrics: metrics.clone(),
            decision_factors: DecisionFactors {
                cpu_threshold_met: false,
                io_threshold_met: false,
                memory_threshold_met: false,
                confidence_threshold_met: recommendation.confidence >= 0.7,
                min_switch_interval_met: false,
                auto_switch_enabled: false,
                different_scheduler: current_scheduler != Some(&recommendation.scheduler_name),
            },
            execution_id: None,
            switch_duration_ms: None,
            success: false,
            error_message: None,
        };

        self.log_switch_event(&event)
    }

    /// Initialize logging system
    pub fn init_logging(&self, log_level: &str) -> Result<()> {
        use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
        use std::path::Path;

        // Extract directory from file path
        let log_path = Path::new(&self.log_file_path);
        let log_dir = if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent)?;
            parent
        } else {
            Path::new(".")
        };

        let file_appender = tracing_appender::rolling::daily(
            log_dir, 
            log_path.file_name().unwrap_or_else(|| std::ffi::OsStr::new("autosa"))
        );
        let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

        let env_filter = EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(log_level));

        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()
                    .with_writer(non_blocking)
                    .with_target(true)
                    .with_thread_ids(true)
            )
            .with(
                tracing_subscriber::fmt::layer()
                    .with_writer(std::io::stdout)
                    .with_target(false)
            )
            .init();

        info!("Logging system initialized, writing to {}", self.log_file_path);
        Ok(())
    }

    /// Get log file path
    pub fn log_file_path(&self) -> &str {
        &self.log_file_path
    }
}

/// Calculate decision factors based on current state and recommendation
pub fn calculate_decision_factors(
    current_scheduler: Option<&str>,
    recommendation: &SchedulerRecommendation,
    config_auto_switch: bool,
    config_min_confidence: f64,
    min_switch_interval_met: bool,
) -> DecisionFactors {
    DecisionFactors {
        cpu_threshold_met: false, // Would need to be calculated based on metrics
        io_threshold_met: false,  // Would need to be calculated based on metrics
        memory_threshold_met: false, // Would need to be calculated based on metrics
        confidence_threshold_met: recommendation.confidence >= config_min_confidence,
        min_switch_interval_met,
        auto_switch_enabled: config_auto_switch,
        different_scheduler: current_scheduler != Some(&recommendation.scheduler_name),
    }
}

/// Create a switching event for a successful switch
pub fn create_switch_event(
    previous_scheduler: Option<String>,
    recommendation: &SchedulerRecommendation,
    metrics: &AggregatedMetrics,
    workload_type: &WorkloadType,
    decision_factors: DecisionFactors,
    execution_id: Option<String>,
    switch_duration_ms: Option<u64>,
) -> SwitchingEvent {
    SwitchingEvent {
        timestamp: Utc::now(),
        previous_scheduler,
        new_scheduler: recommendation.scheduler_name.clone(),
        new_scheduler_args: recommendation.suggested_args.clone(),
        workload_type: workload_type.clone(),
        confidence: recommendation.confidence,
        reason: recommendation.reason.clone(),
        metrics: metrics.clone(),
        decision_factors,
        execution_id,
        switch_duration_ms,
        success: true,
        error_message: None,
    }
}

/// Create a switching event for a failed switch
pub fn create_failed_switch_event(
    previous_scheduler: Option<String>,
    recommendation: &SchedulerRecommendation,
    metrics: &AggregatedMetrics,
    workload_type: &WorkloadType,
    decision_factors: DecisionFactors,
    error_message: String,
) -> SwitchingEvent {
    SwitchingEvent {
        timestamp: Utc::now(),
        previous_scheduler,
        new_scheduler: recommendation.scheduler_name.clone(),
        new_scheduler_args: recommendation.suggested_args.clone(),
        workload_type: workload_type.clone(),
        confidence: recommendation.confidence,
        reason: recommendation.reason.clone(),
        metrics: metrics.clone(),
        decision_factors,
        execution_id: None,
        switch_duration_ms: None,
        success: false,
        error_message: Some(error_message),
    }
}