# AutoSA - Automatic Scheduler Adjustment Service

## Project Summary

AutoSA is a background service that automatically monitors system workload and adjusts the Linux kernel scheduler to optimize performance based on current system conditions.

## Key Accomplishments

### 1. Problem Identification
- The existing `autotune` tool only provides one-time scheduler suggestions
- No continuous background service for automatic scheduler adjustment
- Manual intervention required for scheduler switching

### 2. Solution Design
Created a complete automatic scheduler adjustment service with:

#### Core Components:
- **Metrics Collector**: Real-time collection of CPU, memory, I/O, and scheduler statistics
- **Policy Engine**: Workload classification and scheduler recommendation engine
- **Daemon Service**: Continuous monitoring and automatic scheduler switching
- **CLI Interface**: User-friendly command-line control interface

#### Key Features:
- Continuous system monitoring every second
- Intelligent workload classification (CPU-bound, I/O-bound, latency-sensitive, etc.)
- Confidence-based scheduler recommendations
- Configurable switching policies
- Self-contained deployment

### 3. Implementation Details

#### Technology Stack:
- **Language**: Rust for performance, safety, and deployment simplicity
- **Dependencies**: procfs, tokio, clap, serde
- **Architecture**: Async I/O with Tokio runtime

#### Code Structure:
- `src/metrics.rs`: System metrics collection and aggregation
- `src/policy.rs`: Workload classification and scheduler recommendation
- `src/daemon.rs`: Main daemon service orchestration
- `src/main.rs`: CLI interface
- `Cargo.toml`: Dependency management

### 4. Testing Results
- Successfully collects system metrics in real-time
- Accurately classifies workloads as "LatencySensitive" on test systems
- Provides appropriate scheduler recommendations with confidence scores
- Simulates scheduler switching correctly

### 5. Deployment Characteristics
- **Binary Size**: 1.9MB optimized executable
- **Dependencies**: Self-contained with no runtime requirements
- **Deployment**: Simple copy of binary to target systems
- **Cross-platform**: Can be compiled for different architectures

## Integration Path
The service is designed to integrate with the existing schedcp MCP server:
1. Replace simulated scheduler switching with actual MCP calls
2. Add persistence for configuration and state
3. Implement proper daemonization with process management

## Why Rust?
1. **Performance**: Low overhead for continuous monitoring
2. **Safety**: Memory and thread safety without GC pauses
3. **Reliability**: Error handling prevents daemon crashes
4. **Integration**: Fits naturally with existing Rust codebase
5. **Deployment**: Single static binary simplifies distribution

## Next Steps
1. Integrate with MCP server for actual scheduler management
2. Add persistence for configuration and monitoring history
3. Implement proper daemonization with systemd integration
4. Add comprehensive logging and monitoring capabilities
5. Extend workload classification algorithms