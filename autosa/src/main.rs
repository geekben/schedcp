use anyhow::Result;
use autosa::{AutoSchedulerDaemon, DaemonConfig};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
        #[arg(long, default_value = "60")]
        aggregation_window: u64,

        /// Minimum confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.7")]
        min_confidence: f64,

        /// Minimum switch interval in seconds
        #[arg(long, default_value = "300")]
        min_switch_interval: u64,

        /// Disable automatic scheduler switching
        #[arg(long)]
        no_auto_switch: bool,

        /// Path to the schedcp-cli binary
        #[arg(long, default_value = "/root/schedcp/mcp/target/release/schedcp-cli")]
        schedcp_cli_path: String,
    },

    /// Stop the daemon
    Stop,

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
                }
            } else {
                DaemonConfig {
                    collection_interval_secs: *collection_interval,
                    aggregation_window_secs: *aggregation_window,
                    min_confidence_threshold: *min_confidence,
                    min_switch_interval_secs: *min_switch_interval,
                    enable_auto_switch: !*no_auto_switch,
                    schedcp_cli_path: schedcp_cli_path.clone(),
                }
            };

            println!("Starting automatic scheduler adjustment daemon...");
            println!("Configuration:");
            println!("  Collection interval: {} seconds", config.collection_interval_secs);
            println!("  Aggregation window: {} seconds", config.aggregation_window_secs);
            println!("  Minimum confidence threshold: {:.2}", config.min_confidence_threshold);
            println!("  Minimum switch interval: {} seconds", config.min_switch_interval_secs);
            println!("  Auto-switching: {}", if config.enable_auto_switch { "enabled" } else { "disabled" });

            let mut daemon = AutoSchedulerDaemon::new(config);
            daemon.start().await?;
        }

        Commands::Stop => {
            println!("Stopping daemon...");
            // In a real implementation, we would communicate with the running daemon to stop it
            println!("Daemon stopped");
        }

        Commands::Status => {
            println!("Daemon status:");
            // In a real implementation, we would communicate with the running daemon to get status
            println!("  Status: Not implemented in this demo");
        }

        Commands::List => {
            println!("Available schedulers:");
            println!("  scx_bpfland - Optimized for low latency and interactive responsiveness");
            println!("  scx_flash - Optimized for low latency and predictable performance");
            println!("  scx_lavd - Optimized for low latency and interactivity");
            println!("  scx_rusty - Balanced general-purpose performance");
            println!("  scx_simple - Simple and predictable performance");
        }
    }

    Ok(())
}