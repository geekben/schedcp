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

        /// Log level for tracing (trace, debug, info, warn, error)
        #[arg(long, default_value = "warn")]
        log_level: String,
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
                }
            };

            println!("Starting automatic scheduler adjustment daemon...");
            println!("Configuration:");
            println!("  Collection interval: {} seconds", config.collection_interval_secs);
            println!("  Aggregation window: {} seconds", config.aggregation_window_secs);
            println!("  Minimum confidence threshold: {:.2}", config.min_confidence_threshold);
            println!("  Minimum switch interval: {} seconds", config.min_switch_interval_secs);
            println!("  Auto-switching: {}", if config.enable_auto_switch { "enabled" } else { "disabled" });
            println!("  Log file path: {}", config.log_file_path);
            println!("  Log level: {}", config.log_level);

            // Check if daemon is already running
            let pid_file = "/tmp/autosa.pid";
            if fs::metadata(pid_file).is_ok() {
                let pid_content = fs::read_to_string(&pid_file)?;
                if let Ok(pid) = pid_content.trim().parse::<u32>() {
                    // Check if the process is still running
                    let status = process::Command::new("kill")
                        .arg("-0")
                        .arg(pid.to_string())
                        .status();

                    if let Ok(status) = status {
                        if status.success() {
                            eprintln!("Error: autosa daemon is already running with PID {}", pid);
                            eprintln!("Use 'autosa stop' to stop the existing daemon first");
                            std::process::exit(1);
                        }
                    }
                }
                // PID file exists but process is not running, remove stale file
                let _ = fs::remove_file(&pid_file);
            }

            let mut daemon = AutoSchedulerDaemon::new(config)?;

            // Write PID file for stop command to use
            fs::write(pid_file, process::id().to_string())?;

            // Start the daemon directly
            daemon.start().await?;

            // Clean up PID file on normal exit
            let _ = fs::remove_file(pid_file);
        }

        Commands::Stop { force } => {
            println!("Stopping daemon...");

            // Try to gracefully stop the daemon
            let pid_file = "/tmp/autosa.pid";
            if fs::metadata(pid_file).is_ok() {
                let pid_content = fs::read_to_string(pid_file)?;
                if let Ok(pid) = pid_content.trim().parse::<u32>() {
                    // Send SIGTERM to the daemon
                    if !force {
                        process::Command::new("kill")
                            .arg("-TERM")
                            .arg(pid.to_string())
                            .output()?;

                        // Wait a moment for graceful shutdown
                        std::thread::sleep(std::time::Duration::from_secs(2));
                    }

                    // Check if process is still running
                    let status = process::Command::new("kill")
                        .arg("-0")
                        .arg(pid.to_string())
                        .status()?;

                    if status.success() && *force {
                        // Force kill if still running and force flag is set
                        process::Command::new("kill")
                            .arg("-KILL")
                            .arg(pid.to_string())
                            .output()?;
                    }

                    // Remove PID file
                    let _ = fs::remove_file(pid_file);

                    // Clean up any remaining .tmp directories
                    println!("Cleaning up any remaining temporary directories...");
                    let output = process::Command::new("find")
                        .args(&["/tmp", "-name", ".tmp*", "-type", "d"])
                        .output()?;

                    if output.status.success() {
                        let temp_dirs = String::from_utf8_lossy(&output.stdout);
                        for line in temp_dirs.lines() {
                            let path = line.trim();
                            if !path.is_empty() {
                                let _ = process::Command::new("rm")
                                    .arg("-rf")
                                    .arg(path)
                                    .output();
                                println!("Cleaned up temporary directory: {}", path);
                            }
                        }
                    }

                    println!("Daemon stopped");
                } else {
                    eprintln!("Invalid PID in pid file");
                    // Remove stale PID file
                    let _ = fs::remove_file(pid_file);
                }
            } else {
                println!("Daemon is not running (no PID file found)");
            }
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
