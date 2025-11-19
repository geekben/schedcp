use anyhow::Result;
use autosa::{AutoSchedulerDaemon, DaemonConfig};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::fs;
use std::process;

#[derive(Parser)]
#[command(name = "autosa")]
#[command(about = "Automatic Scheduler Adjustment - Automatically optimize Linux kernel schedulers based on workload", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the automatic scheduler adjustment daemon
    Start {
        /// Configuration file path
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Collection interval in seconds
        #[arg(long, default_value = "1")]
        collection_interval: u64,

        /// Aggregation window in seconds
        #[arg(long, default_value = "3")]
        aggregation_window: u64,

        /// Minimum confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.7")]
        min_confidence: f64,

        /// Minimum switch interval in seconds
        #[arg(long, default_value = "9")]
        min_switch_interval: u64,

        /// Disable automatic scheduler switching
        #[arg(long)]
        no_auto_switch: bool,

        /// Path to the schedcp-cli binary
        #[arg(long, default_value = "/root/schedcp/mcp/target/release/schedcp-cli")]
        schedcp_cli_path: String,

        /// Path to the switching events log file
        #[arg(long, default_value = "/var/log/autosa/switching_events.log")]
        log_file_path: String,

        /// Log level (trace, debug, info, warn, error)
        #[arg(long, default_value = "warn")]
        log_level: String,

        /// Path to iFlow CLI binary
        #[arg(long, default_value = "iflow")]
        iflow_cli_path: String,

        /// iFlow model name
        #[arg(long, default_value = "default")]
        iflow_model: String,

        /// Disable AI-based scheduler selection
        #[arg(long)]
        no_ai_selection: bool,

        /// AI confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.8")]
        ai_confidence_threshold: f64,

        /// AI call interval in seconds
        #[arg(long, default_value = "30")]
        ai_call_interval: u64,

        /// Maximum AI calls per hour
        #[arg(long, default_value = "60")]
        ai_max_calls_per_hour: u32,

        /// AI cache duration in seconds
        #[arg(long, default_value = "300")]
        ai_cache_duration: u64,

        /// Minimum metric change to trigger AI (percentage)
        #[arg(long, default_value = "0.15")]
        ai_min_change_threshold: f64,

        /// Performance degradation threshold for rollback (0.0-1.0)
        #[arg(long, default_value = "0.15")]
        performance_degradation_threshold: f64,

        /// Minimum samples for performance evaluation
        #[arg(long, default_value = "5")]
        min_performance_samples: u32,

        /// Performance trend sensitivity (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        performance_trend_sensitivity: f64,

        /// Hardware metrics weight in performance score (0.0-1.0)
        #[arg(long, default_value = "0.4")]
        hardware_metrics_weight: f64,
    },

    /// Stop the daemon
    Stop {
        /// Force stop even if daemon doesn't respond
        #[arg(long)]
        force: bool,
    },

    /// Get current daemon status
    Status,

    /// List available schedulers
    List,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Start {
            config,
            collection_interval,
            aggregation_window,
            min_confidence,
            min_switch_interval,
            no_auto_switch,
            schedcp_cli_path,
            log_file_path,
            log_level,
            iflow_cli_path,
            iflow_model,
            no_ai_selection,
            ai_confidence_threshold,
            ai_call_interval,
            ai_max_calls_per_hour,
            ai_cache_duration,
            ai_min_change_threshold,
            performance_degradation_threshold,
            min_performance_samples,
            performance_trend_sensitivity,
            hardware_metrics_weight,
        } => {
            // Load configuration
            let config = if let Some(_config_path) = config {
                // In a real implementation, we would load from file
                // For now, we'll use the provided command line arguments
                DaemonConfig {
                    collection_interval_secs: *collection_interval,
                    aggregation_window_secs: *aggregation_window,
                    min_confidence_threshold: *min_confidence,
                    min_switch_interval_secs: *min_switch_interval,
                    enable_auto_switch: !*no_auto_switch,
                    schedcp_cli_path: schedcp_cli_path.clone(),
                    log_file_path: log_file_path.clone(),
                    log_level: log_level.clone(),
                    system_profile: Default::default(),
                    enable_performance_feedback: true,
                    stabilization_period_secs: 30,
                    min_performance_improvement: 0.1,
                    iflow_cli_path: iflow_cli_path.clone(),
                    iflow_model_name: iflow_model.clone(),
                    enable_ai_selection: !*no_ai_selection,
                    ai_confidence_threshold: *ai_confidence_threshold,
                    ai_call_interval_secs: *ai_call_interval,
                    ai_max_calls_per_hour: *ai_max_calls_per_hour,
                    ai_cache_duration_secs: *ai_cache_duration,
                    ai_min_change_threshold: *ai_min_change_threshold,
                    performance_degradation_threshold: *performance_degradation_threshold,
                    min_performance_samples: *min_performance_samples,
                    performance_trend_sensitivity: *performance_trend_sensitivity,
                    hardware_metrics_weight: *hardware_metrics_weight,
                }
            } else {
                DaemonConfig {
                    collection_interval_secs: *collection_interval,
                    aggregation_window_secs: *aggregation_window,
                    min_confidence_threshold: *min_confidence,
                    min_switch_interval_secs: *min_switch_interval,
                    enable_auto_switch: !*no_auto_switch,
                    schedcp_cli_path: schedcp_cli_path.clone(),
                    log_file_path: log_file_path.clone(),
                    log_level: log_level.clone(),
                    system_profile: Default::default(),
                    enable_performance_feedback: true,
                    stabilization_period_secs: 30,
                    min_performance_improvement: 0.1,
                    iflow_cli_path: iflow_cli_path.clone(),
                    iflow_model_name: iflow_model.clone(),
                    enable_ai_selection: !*no_ai_selection,
                    ai_confidence_threshold: *ai_confidence_threshold,
                    ai_call_interval_secs: *ai_call_interval,
                    ai_max_calls_per_hour: *ai_max_calls_per_hour,
                    ai_cache_duration_secs: *ai_cache_duration,
                    ai_min_change_threshold: *ai_min_change_threshold,
                    performance_degradation_threshold: *performance_degradation_threshold,
                    min_performance_samples: *min_performance_samples,
                    performance_trend_sensitivity: *performance_trend_sensitivity,
                    hardware_metrics_weight: *hardware_metrics_weight,
                }
            };

            println!("Starting automatic scheduler adjustment daemon...");
            println!("Configuration:");
            println!("  Collection interval: {} seconds", config.collection_interval_secs);
            println!("  Aggregation window: {} seconds", config.aggregation_window_secs);
            println!("  Min confidence threshold: {:.2}", config.min_confidence_threshold);
            println!("  Min switch interval: {} seconds", config.min_switch_interval_secs);
            println!("  Auto-switch: {}", config.enable_auto_switch);
            println!("  AI selection: {}", config.enable_ai_selection);
            if config.enable_ai_selection {
                println!("  AI confidence threshold: {:.2}", config.ai_confidence_threshold);
            }
            println!("  Log file: {}", config.log_file_path);
            println!("  Log level: {}", config.log_level);

            // Create log directory if it doesn't exist
            if let Some(parent) = PathBuf::from(&config.log_file_path).parent() {
                fs::create_dir_all(parent)?;
            }

            // Create and start daemon
            let mut daemon = AutoSchedulerDaemon::new(config)?;

            // Detect system profile
            daemon.detect_system_profile();

            // Start the daemon
            daemon.start().await?;
        }

        Commands::Stop { force } => {
            println!("Stopping automatic scheduler adjustment daemon...");
            // Implementation would connect to running daemon and send stop signal
            if *force {
                println!("Force stopping daemon...");
            }
        }

        Commands::Status => {
            println!("Getting daemon status...");
            // Implementation would query daemon status
        }

        Commands::List => {
            println!("Available schedulers:");
            println!("  scx_bpfland - Interactive workloads, low latency");
            println!("  scx_flash - Low latency and predictable performance");
            println!("  scx_lavd - Low latency and interactivity");
            println!("  scx_rusty - Balanced general-purpose performance");
            println!("  scx_simple - Simple and predictable performance");
            println!("  disable - Use system default scheduler (disable sched_ext)");
        }
    }

    Ok(())
}
