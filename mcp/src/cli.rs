use anyhow::Result;
use clap::{Parser, Subcommand};
use std::env;
use std::collections::HashMap;

mod scheduler_manager;
mod scheduler_generator;
mod system_monitor;
use scheduler_manager::{SchedulerManager, SchedulerInfo};
use system_monitor::SystemMonitor;

#[derive(Parser)]
#[command(name = "schedcp-cli")]
#[command(about = "SchedCP CLI - Run and manage sched-ext schedulers", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List schedulers with detailed information
    List {
        /// Filter by scheduler name
        #[arg(short, long)]
        name: Option<String>,
        /// Show only production-ready schedulers
        #[arg(short, long)]
        production: bool,
    },
    /// Run a scheduler
    Run {
        /// Name of the scheduler to run
        scheduler: String,
        /// Additional arguments to pass to the scheduler
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
        /// Use sudo to run the scheduler (will prompt for password via SCHEDCP_SUDO_PASSWORD env var)
        #[arg(short, long)]
        sudo: bool,
    },
    /// Create, compile, and run a custom scheduler from source code
    CreateAndRun {
        /// Path to the BPF C source file (.bpf.c)
        source: String,
    },
    /// Monitor system metrics (CPU, memory, scheduler) for a specified duration
    Monitor {
        /// Duration in seconds to monitor the system
        #[arg(short, long, default_value = "10")]
        duration: u64,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut manager = SchedulerManager::new()?;

    match cli.command {
        Commands::List { name, production } => {
            let schedulers = manager.list_schedulers();

            for scheduler in schedulers {
                // Apply filters
                if production && !scheduler.production_ready {
                    continue;
                }
                if let Some(ref filter_name) = name {
                    if !scheduler.name.contains(filter_name) {
                        continue;
                    }
                }

                // Print detailed information for each scheduler
                manager.print_scheduler_info(scheduler);
            }
        }

        Commands::Run { scheduler, args, sudo } => {
            // Extract schedulers to temporary directory
            manager.extract_schedulers().await?;

            // Get sudo password from environment
            // If --sudo flag is set or no flag (default), use sudo
            // Empty string means passwordless sudo
            let sudo_password = if sudo {
                env::var("SCHEDCP_SUDO_PASSWORD").ok().unwrap_or_default()
            } else {
                // Even without --sudo flag, still use passwordless sudo for schedulers
                String::new()
            };

            manager.set_sudo_password(sudo_password);

            // Run the scheduler using create_execution API (handles both built-in and custom)
            println!("Starting scheduler: {}", scheduler);
            let execution_id = manager.create_execution(&scheduler, args).await?;
            println!("Scheduler started with execution ID: {}", execution_id);

            // Keep the process running until interrupted
            println!("\nPress Ctrl+C to stop the scheduler...");
            tokio::signal::ctrl_c().await?;

            println!("\nStopping scheduler...");
            manager.stop_scheduler(&execution_id).await?;
            println!("Scheduler stopped");

            // Clean up temporary directories
            manager.cleanup();
        }

        Commands::CreateAndRun { source } => {
            // Set sudo password from environment
            let sudo_password = env::var("SCHEDCP_SUDO_PASSWORD").ok();
            if let Some(ref password) = sudo_password {
                manager.set_sudo_password(password.clone());
            }

            // Infer scheduler name from source file (e.g., "my_sched.bpf.c" -> "my_sched")
            let source_path = std::path::Path::new(&source);
            let name = source_path
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.trim_end_matches(".bpf"))
                .ok_or_else(|| anyhow::anyhow!("Invalid source file name"))?
                .to_string();

            // Read source code from file
            println!("Reading source code from: {}", source);
            let source_code = std::fs::read_to_string(&source)?;

            // Create minimal scheduler info
            let info = SchedulerInfo {
                name: name.clone(),
                production_ready: false,
                description: format!("Custom scheduler compiled from {}", source),
                use_cases: vec!["Custom workload".to_string()],
                algorithm: "Custom".to_string(),
                characteristics: "User-provided BPF scheduler".to_string(),
                tuning_parameters: HashMap::new(),
                limitations: "Not verified for production use".to_string(),
                performance_profile: "Unknown".to_string(),
            };

            println!("Creating and verifying custom scheduler: {}", name);

            // Create and verify the scheduler through the manager
            let result = manager.create_and_verify_scheduler(info, &source_code).await?;
            println!("{}", result);

            println!("\nStarting custom scheduler...");
            let execution_id = manager.create_execution(&name, vec![]).await?;
            println!("Custom scheduler started with execution ID: {}", execution_id);

            // Keep the process running until interrupted
            println!("\nPress Ctrl+C to stop the scheduler...");
            tokio::signal::ctrl_c().await?;

            println!("\nStopping scheduler...");
            manager.stop_scheduler(&execution_id).await?;
            println!("Scheduler stopped");

            // Clean up temporary directories
            manager.cleanup();
        }

        Commands::Monitor { duration } => {
            println!("Starting system monitoring for {} seconds...", duration);
            println!("Collecting CPU, memory, and scheduler metrics every second.\n");

            let monitor = SystemMonitor::new();

            // Start monitoring
            let session_id = monitor.start_monitoring().await?;
            println!("Monitoring session started: {}\n", session_id);

            // Wait for the specified duration
            tokio::time::sleep(tokio::time::Duration::from_secs(duration)).await;

            // Stop and get summary
            println!("Stopping monitoring...\n");
            let summary = monitor.stop_monitoring().await?;

            // Display summary
            println!("=== System Monitoring Summary ===");
            println!("Session ID: {}", summary.session_id);
            println!("Duration: {} seconds", summary.duration_secs);
            println!("Samples collected: {}", summary.sample_count);
            println!();
            println!("CPU Utilization:");
            println!("  Average: {:.2}%", summary.cpu_avg_percent);
            println!("  Maximum: {:.2}%", summary.cpu_max_percent);
            println!();
            println!("Memory Usage:");
            println!("  Average: {:.2}%", summary.memory_avg_percent);
            println!("  Maximum: {:.2}%", summary.memory_max_percent);
            println!("  Average Used: {:.2} MB", summary.memory_avg_used_mb);
            println!();
            println!("Scheduler Statistics:");
            println!("  Total Timeslices: {}", summary.sched_total_timeslices);
            println!("  Avg Run Time: {} ns", summary.sched_avg_run_time_ns);
            println!();
        }
    }

    // Clean up temporary directories for all commands
    manager.cleanup();

    Ok(())
}