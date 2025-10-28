//! AutoSA - Automatic Scheduler Adjustment
//!
//! A service that automatically monitors system workload and adjusts the Linux kernel scheduler
//! to optimize performance based on current system conditions.

pub mod metrics;
pub mod policy;
pub mod daemon;
pub mod mcp_client;

// Re-export the main types for convenience
pub use daemon::{AutoSchedulerDaemon, DaemonConfig, DaemonState};
pub use metrics::{AggregatedMetrics, MetricsCollector, SystemMetrics};
pub use mcp_client::McpClient;
pub use policy::{PolicyEngine, SchedulerRecommendation, WorkloadType};

use anyhow::Result;

/// Start the automatic scheduler adjustment daemon with default configuration
pub async fn start_default_daemon() -> Result<()> {
    let config = DaemonConfig::default();
    let mut daemon = AutoSchedulerDaemon::new(config);
    daemon.start().await
}