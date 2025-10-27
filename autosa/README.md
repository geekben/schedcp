# AutoSA - Automatic Scheduler Adjustment

AutoSA is a background service that automatically monitors system workload and adjusts the Linux kernel scheduler to optimize performance based on current system conditions.

## Quick Summary

AutoSA solves the problem that the existing `autotune` tool only provides one-time scheduler suggestions. It provides continuous, automatic scheduler adjustment based on real-time workload analysis. The service:

- Monitors system metrics every second (CPU, memory, I/O, scheduler stats)
- Classifies workloads in real-time (CPU-bound, I/O-bound, latency-sensitive, etc.)
- Recommends and automatically switches schedulers based on workload characteristics
- Deploys as a single 1.9MB binary with no runtime dependencies
- Built in Rust for performance, safety, and reliability

See [SUMMARY.md](SUMMARY.md) for complete implementation details and [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) for production deployment guidelines.

## Overview

AutoSA continuously monitors system metrics including:
- CPU utilization
- Memory usage
- I/O statistics
- Scheduler performance metrics

Based on these metrics, it classifies the current workload type and recommends the most appropriate scheduler. When configured to do so, it can automatically switch between different schedulers to optimize system performance.

## Features

- **Continuous Monitoring**: Collects system metrics every second
- **Workload Classification**: Automatically identifies workload types (CPU-bound, I/O-bound, latency-sensitive, etc.)
- **Intelligent Scheduler Selection**: Recommends optimal schedulers based on workload characteristics
- **Automatic Switching**: Can automatically switch schedulers when workload patterns change
- **Configurable Policies**: Adjustable parameters for switching behavior
- **Integration Ready**: Designed to integrate with the schedcp MCP server

## Supported Schedulers

AutoSA works with the following schedulers from the sched-ext framework:

- **scx_bpfland**: Optimized for low latency and interactive responsiveness
- **scx_flash**: Optimized for low latency and predictable performance
- **scx_lavd**: Optimized for low latency and interactivity
- **scx_rusty**: Balanced general-purpose performance
- **scx_simple**: Simple and predictable performance

## Installation

Build from source:

```bash
cargo build --release
```

The binary will be available at `target/release/autosa`.

## Usage

### Start the daemon

```bash
# Start with default settings
autosa start

# Start with custom configuration
autosa start --collection-interval 2 --aggregation-window 120 --min-confidence 0.8
```

### Stop the daemon

```bash
autosa stop
```

### View daemon status

```bash
autosa status
```

### List available schedulers

```bash
autosa list
```

## Configuration

AutoSA can be configured with the following parameters:

- `collection_interval_secs`: How often to collect metrics (default: 1 second)
- `aggregation_window_secs`: Time window for calculating aggregated metrics (default: 60 seconds)
- `min_confidence_threshold`: Minimum confidence level required to switch schedulers (default: 0.7)
- `min_switch_interval_secs`: Minimum time between scheduler switches (default: 300 seconds)
- `enable_auto_switch`: Enable automatic scheduler switching (default: true)

## Architecture

AutoSA consists of several key components:

1. **Metrics Collector**: Gathers system metrics from `/proc` filesystem
2. **Policy Engine**: Analyzes metrics and classifies workloads
3. **Daemon Service**: Orchestrates monitoring and scheduler switching
4. **CLI Interface**: Command-line interface for controlling the daemon

## Integration

AutoSA is designed to integrate with the schedcp MCP server for actual scheduler management. The current implementation includes a simulation layer that can be replaced with real MCP calls.

## Requirements

- Linux kernel 6.12+ with sched-ext support
- Rust 1.82+ for building
- Root privileges for scheduler management (in integrated version)

## Deployment

### Simple Deployment
The AutoSA binary is self-contained and can be deployed by simply copying it:
```bash
# Copy the binary to the target machine
cp target/release/autosa /usr/local/bin/
chmod +x /usr/local/bin/autosa
```

### Complete Deployment (Production)
For production use, you might want to include proper service management:

```bash
# Copy binary
sudo cp target/release/autosa /usr/local/bin/

# Create systemd service
sudo tee /etc/systemd/system/autosa.service > /dev/null << EOF
[Unit]
Description=AutoSA - Automatic Scheduler Adjustment
After=multi-user.target

[Service]
Type=simple
ExecStart=/usr/local/bin/autosa start
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable autosa
sudo systemctl start autosa

# Check service status
sudo systemctl status autosa

# View logs
sudo journalctl -u autosa -f
```

### Requirements for Target Machines
- Same CPU architecture as the build machine
- Linux kernel 6.12+ with sched-ext support
- Root privileges for actual scheduler management

### Cross-Platform Builds
For different architectures, recompile with the appropriate target:
```bash
# For ARM64
cargo build --release --target aarch64-unknown-linux-gnu

# For older x86 systems
cargo build --release --target i686-unknown-linux-gnu
```

### Verification
After deployment, verify the service is working:
```bash
# Check version and help
autosa --help

# Test with short run
timeout 10s autosa start --collection-interval 1 --aggregation-window 5

# Check system resources usage
ps aux | grep autosa
top -p $(pgrep autosa)
```

## Why Rust?

AutoSA uses Rust for several key reasons:

1. **Performance**: Low overhead monitoring with minimal system impact
2. **Safety**: Memory and thread safety without garbage collection pauses
3. **Reliability**: Error handling that prevents daemon crashes
4. **Integration**: Fits naturally with the existing schedcp Rust codebase
5. **Deployment**: Single static binary with no runtime dependencies

## Binary Size Explanation

- **124MB deps directory**: Contains source code, documentation, and build artifacts for all dependencies
- **1.9MB binary**: Optimized, self-contained executable with dead code elimination
- This is normal for Rust projects - the large deps directory enables development while the small binary enables deployment

## License

GPL-2.0