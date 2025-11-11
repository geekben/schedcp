use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::metrics::AggregatedMetrics;

/// Workload type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    CpuBound,
    IoBound,
    Mixed,
    LatencySensitive,
    Batch,
    // New intermediate states for more granular classification
    CpuIntensive,
    IoIntensive,
    MemoryIntensive,
    Transitioning,
}

/// Scheduler recommendation with confidence level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerRecommendation {
    pub scheduler_name: String,
    pub confidence: f64, // 0.0 to 1.0
    pub reason: String,
    pub suggested_args: Vec<String>,
}

/// Policy engine that analyzes metrics and recommends schedulers
pub struct PolicyEngine {
    scheduler_characteristics: HashMap<String, SchedulerInfo>,
    // Add adaptive thresholds based on system baseline
    baseline_metrics: Option<AggregatedMetrics>,
    // Track previous workload for hysteresis
    previous_workload: Option<WorkloadType>,
    // Track workload stability
    workload_history: Vec<WorkloadType>,
    max_history_size: usize,
}

/// Information about a scheduler's characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerInfo {
    pub name: String,
    pub workload_types: Vec<WorkloadType>,
    pub performance_profile: String,
    pub use_cases: Vec<String>,
    pub default_args: Vec<String>,
}

impl PolicyEngine {
    pub fn new() -> Self {
        let mut scheduler_characteristics = HashMap::new();

        // Define characteristics for each scheduler based on the schedulers.json
        scheduler_characteristics.insert("scx_bpfland".to_string(), SchedulerInfo {
            name: "scx_bpfland".to_string(),
            workload_types: vec![WorkloadType::LatencySensitive, WorkloadType::Mixed, WorkloadType::IoIntensive],
            performance_profile: "optimized for low latency and interactive responsiveness".to_string(),
            use_cases: vec!["gaming".to_string(), "live_streaming".to_string(), "multimedia".to_string(), "real_time_audio".to_string()],
            default_args: vec!["--slice-us".to_string(), "20000".to_string()],
        });

        scheduler_characteristics.insert("scx_flash".to_string(), SchedulerInfo {
            name: "scx_flash".to_string(),
            workload_types: vec![WorkloadType::LatencySensitive, WorkloadType::Mixed, WorkloadType::Transitioning],
            performance_profile: "optimized for low latency and predictable performance".to_string(),
            use_cases: vec!["multimedia".to_string(), "real_time_audio".to_string(), "predictable_performance".to_string(), "latency_sensitive".to_string()],
            default_args: vec!["--slice-us".to_string(), "4096".to_string()],
        });

        scheduler_characteristics.insert("scx_lavd".to_string(), SchedulerInfo {
            name: "scx_lavd".to_string(),
            workload_types: vec![WorkloadType::LatencySensitive, WorkloadType::Mixed, WorkloadType::CpuIntensive],
            performance_profile: "optimized for low latency and interactivity".to_string(),
            use_cases: vec!["gaming".to_string(), "interactive_applications".to_string(), "low_latency".to_string()],
            default_args: vec!["--slice-max-us".to_string(), "5000".to_string()],
        });

        scheduler_characteristics.insert("scx_rusty".to_string(), SchedulerInfo {
            name: "scx_rusty".to_string(),
            workload_types: vec![WorkloadType::Batch, WorkloadType::Mixed, WorkloadType::CpuBound, WorkloadType::MemoryIntensive],
            performance_profile: "balanced general-purpose performance".to_string(),
            use_cases: vec!["general_purpose".to_string(), "multi_architecture".to_string(), "server_workloads".to_string()],
            default_args: vec!["--slice-us-underutil".to_string(), "20000".to_string()],
        });

        scheduler_characteristics.insert("scx_simple".to_string(), SchedulerInfo {
            name: "scx_simple".to_string(),
            workload_types: vec![WorkloadType::Batch, WorkloadType::Mixed],
            performance_profile: "simple and predictable performance".to_string(),
            use_cases: vec!["single_socket".to_string(), "uniform_l3_cache".to_string(), "simple_workloads".to_string()],
            default_args: vec!["--slice-us".to_string(), "20000".to_string()],
        });

        Self {
            scheduler_characteristics,
            baseline_metrics: None,
            previous_workload: None,
            workload_history: Vec::new(),
            max_history_size: 10,
        }
    }

    /// Classify the current workload based on aggregated metrics
    pub fn classify_workload(&mut self, metrics: &super::metrics::AggregatedMetrics) -> WorkloadType {
        // Initialize baseline if not set
        if self.baseline_metrics.is_none() {
            self.baseline_metrics = Some(metrics.clone());
        }

        // Get adaptive thresholds based on system baseline
        let (cpu_high, cpu_low, io_high, io_low, memory_high) = self.get_adaptive_thresholds(metrics);

        // More nuanced classification with intermediate states
        let current_workload = if metrics.cpu_avg_percent > cpu_high && metrics.cpu_iowait_percent < io_low {
            if metrics.cpu_avg_percent > cpu_high + 10.0 {
                WorkloadType::CpuBound
            } else {
                WorkloadType::CpuIntensive
            }
        } else if metrics.cpu_iowait_percent > io_high {
            if metrics.cpu_iowait_percent > io_high + 10.0 {
                WorkloadType::IoBound
            } else {
                WorkloadType::IoIntensive
            }
        } else if metrics.memory_avg_percent > memory_high && metrics.cpu_avg_percent > cpu_low {
            WorkloadType::MemoryIntensive
        } else if metrics.cpu_avg_percent < cpu_low && metrics.cpu_iowait_percent < io_low {
            WorkloadType::LatencySensitive
        } else if metrics.cpu_avg_percent < cpu_low + 20.0 && metrics.cpu_iowait_percent < io_low + 10.0 {
            WorkloadType::Batch
        } else {
            WorkloadType::Mixed
        };

        // Apply hysteresis to prevent oscillation
        let stabilized_workload = self.apply_hysteresis(current_workload);

        // Update history
        self.update_workload_history(stabilized_workload.clone());

        stabilized_workload
    }

    /// Get adaptive thresholds based on system baseline and current metrics
    fn get_adaptive_thresholds(&self, metrics: &super::metrics::AggregatedMetrics) -> (f64, f64, f64, f64, f64) {
        // Default thresholds
        let mut cpu_high = 70.0;
        let mut cpu_low = 30.0;
        let mut io_high = 20.0;
        let mut io_low = 10.0;
        let mut memory_high = 80.0;

        // Adjust based on baseline if available
        if let Some(baseline) = &self.baseline_metrics {
            // Adjust CPU thresholds based on baseline CPU usage
            if baseline.cpu_avg_percent > 50.0 {
                // System typically has high CPU usage, adjust thresholds
                cpu_high = baseline.cpu_avg_percent + 20.0;
                cpu_low = baseline.cpu_avg_percent - 20.0;
            }

            // Adjust I/O thresholds based on baseline I/O
            if baseline.cpu_iowait_percent > 15.0 {
                // System typically has high I/O wait
                io_high = baseline.cpu_iowait_percent + 10.0;
                io_low = baseline.cpu_iowait_percent - 5.0;
            }

            // Adjust memory thresholds based on baseline memory
            if baseline.memory_avg_percent > 70.0 {
                memory_high = baseline.memory_avg_percent + 10.0;
            }
        }

        // Ensure thresholds stay within reasonable bounds
        cpu_high = cpu_high.clamp(60.0, 95.0);
        cpu_low = cpu_low.clamp(10.0, 50.0);
        io_high = io_high.clamp(15.0, 40.0);
        io_low = io_low.clamp(5.0, 20.0);
        memory_high = memory_high.clamp(70.0, 95.0);

        (cpu_high, cpu_low, io_high, io_low, memory_high)
    }

    /// Apply hysteresis to prevent workload oscillation
    fn apply_hysteresis(&mut self, current_workload: WorkloadType) -> WorkloadType {
        // If this is the first classification, use it directly
        if let Some(ref previous) = self.previous_workload {
            // Check if we're in a transition state
            if previous != &current_workload {
                // Check if we have enough history to confirm the transition
                if self.workload_history.len() >= 3 {
                    let recent_count = self.workload_history.iter()
                        .rev()
                        .take(3)
                        .filter(|&w| w == &current_workload)
                        .count();

                    // Only switch if we have consistent classification
                    if recent_count >= 2 {
                        self.previous_workload = Some(current_workload.clone());
                        return current_workload;
                    } else {
                        // Stay in transitioning state temporarily
                        return WorkloadType::Transitioning;
                    }
                }
            }
        }

        self.previous_workload = Some(current_workload.clone());
        current_workload
    }

    /// Update workload history for stability analysis
    fn update_workload_history(&mut self, workload: WorkloadType) {
        self.workload_history.push(workload);
        if self.workload_history.len() > self.max_history_size {
            self.workload_history.remove(0);
        }
    }

    /// Check if current workload is stable
    pub fn is_workload_stable(&self) -> bool {
        if self.workload_history.len() < 5 {
            return false;
        }

        let recent_workloads: std::collections::HashSet<_> = self.workload_history.iter()
            .rev()
            .take(5)
            .collect();

        recent_workloads.len() <= 2 // Allow at most 2 different workload types
    }

    /// Recommend a scheduler based on workload classification and metrics
    pub fn recommend_scheduler(&mut self, metrics: &super::metrics::AggregatedMetrics) -> SchedulerRecommendation {
        let workload_type = self.classify_workload(metrics);

        // Match workload type to appropriate scheduler
        let (scheduler_name, reason) = match workload_type {
            WorkloadType::CpuBound => {
                ("scx_rusty", "High CPU utilization detected, scx_rusty provides good throughput for CPU-bound workloads")
            },
            WorkloadType::CpuIntensive => {
                ("scx_lavd", "Moderate-high CPU usage detected, scx_lavd handles CPU-intensive interactive workloads well")
            },
            WorkloadType::IoBound => {
                ("scx_bpfland", "High I/O wait detected, scx_bpfland prioritizes interactive tasks")
            },
            WorkloadType::IoIntensive => {
                ("scx_bpfland", "Moderate I/O wait detected, scx_bpfland provides good I/O responsiveness")
            },
            WorkloadType::MemoryIntensive => {
                ("scx_rusty", "High memory usage with moderate CPU detected, scx_rusty handles memory-intensive workloads well")
            },
            WorkloadType::LatencySensitive => {
                ("scx_lavd", "Low CPU usage with low I/O wait suggests latency-sensitive workload, scx_lavd is optimized for this")
            },
            WorkloadType::Batch => {
                ("scx_rusty", "Stable workload pattern suggests batch processing, scx_rusty provides balanced performance")
            },
            WorkloadType::Mixed => {
                ("scx_flash", "Mixed workload characteristics detected, scx_flash provides balanced performance with low latency")
            },
            WorkloadType::Transitioning => {
                // During transition, prefer staying with current or using a conservative scheduler
                ("scx_rusty", "Workload transitioning detected, using conservative scheduler for stability")
            },
        };

        // Calculate confidence based on how clearly the metrics match the classification
        let confidence = self.calculate_confidence(metrics, &workload_type);

        // Get suggested arguments for the recommended scheduler
        let suggested_args = if let Some(scheduler_info) = self.scheduler_characteristics.get(scheduler_name) {
            scheduler_info.default_args.clone()
        } else {
            vec![]
        };

        SchedulerRecommendation {
            scheduler_name: scheduler_name.to_string(),
            confidence,
            reason: reason.to_string(),
            suggested_args,
        }
    }

    /// Calculate confidence level based on how clearly the metrics match the workload classification
    fn calculate_confidence(&self, metrics: &super::metrics::AggregatedMetrics, workload_type: &WorkloadType) -> f64 {
        let base_confidence = match workload_type {
            WorkloadType::CpuBound => {
                // Confidence based on high CPU usage and low I/O wait
                let cpu_confidence = self.normalize_confidence(metrics.cpu_avg_percent, 70.0, 95.0);
                let io_confidence = self.normalize_confidence(100.0 - metrics.cpu_iowait_percent, 90.0, 100.0);
                let cpu_max_confidence = self.normalize_confidence(metrics.cpu_max_percent, 75.0, 95.0);
                (cpu_confidence * 0.4 + io_confidence * 0.3 + cpu_max_confidence * 0.3)
            },
            WorkloadType::CpuIntensive => {
                // Moderate-high CPU usage
                let cpu_confidence = self.normalize_confidence(metrics.cpu_avg_percent, 50.0, 70.0);
                let io_confidence = self.normalize_confidence(100.0 - metrics.cpu_iowait_percent, 80.0, 95.0);
                (cpu_confidence * 0.6 + io_confidence * 0.4)
            },
            WorkloadType::IoBound => {
                // Confidence based on high I/O wait
                let io_confidence = self.normalize_confidence(metrics.cpu_iowait_percent, 20.0, 40.0);
                let cpu_confidence = self.normalize_confidence(100.0 - metrics.cpu_avg_percent, 50.0, 80.0);
                let io_queue_confidence = self.normalize_confidence(metrics.io_avg_queue_depth, 1.0, 10.0);
                (io_confidence * 0.5 + cpu_confidence * 0.3 + io_queue_confidence * 0.2)
            },
            WorkloadType::IoIntensive => {
                // Moderate I/O wait
                let io_confidence = self.normalize_confidence(metrics.cpu_iowait_percent, 10.0, 20.0);
                let io_queue_confidence = self.normalize_confidence(metrics.io_avg_queue_depth, 0.5, 5.0);
                (io_confidence * 0.6 + io_queue_confidence * 0.4)
            },
            WorkloadType::MemoryIntensive => {
                // High memory usage with moderate CPU
                let mem_confidence = self.normalize_confidence(metrics.memory_avg_percent, 80.0, 95.0);
                let cpu_confidence = self.normalize_confidence(metrics.cpu_avg_percent, 30.0, 60.0);
                let mem_max_confidence = self.normalize_confidence(metrics.memory_max_percent, 85.0, 95.0);
                (mem_confidence * 0.5 + cpu_confidence * 0.3 + mem_max_confidence * 0.2)
            },
            WorkloadType::LatencySensitive => {
                // Confidence based on low CPU and I/O wait
                let cpu_confidence = self.normalize_confidence(100.0 - metrics.cpu_avg_percent, 70.0, 90.0);
                let io_confidence = self.normalize_confidence(100.0 - metrics.cpu_iowait_percent, 85.0, 95.0);
                let timeslice_confidence = self.normalize_confidence(metrics.sched_timeslices_per_sec, 100.0, 1000.0);
                (cpu_confidence * 0.4 + io_confidence * 0.4 + timeslice_confidence * 0.2)
            },
            WorkloadType::Batch => {
                // Stable workload with moderate metrics
                let cpu_stability = 1.0 - (metrics.cpu_max_percent - metrics.cpu_avg_percent) / 100.0;
                let mem_stability = 1.0 - (metrics.memory_max_percent - metrics.memory_avg_percent) / 100.0;
                let cpu_confidence = self.normalize_confidence(metrics.cpu_avg_percent, 20.0, 50.0);
                (cpu_confidence * 0.5 + cpu_stability * 0.25 + mem_stability * 0.25)
            },
            WorkloadType::Mixed => {
                // Balanced metrics across the board
                let cpu_balance = 1.0 - (metrics.cpu_avg_percent - 50.0).abs() / 50.0;
                let io_balance = 1.0 - (metrics.cpu_iowait_percent - 10.0).abs() / 20.0;
                let mem_balance = 1.0 - (metrics.memory_avg_percent - 60.0).abs() / 40.0;
                (cpu_balance * 0.4 + io_balance * 0.3 + mem_balance * 0.3)
            },
            WorkloadType::Transitioning => {
                // Low confidence during transitions
                0.4
            },
        };

        // Apply stability bonus if workload is stable
        let stability_bonus = if self.is_workload_stable() { 0.1 } else { 0.0 };

        // Apply hysteresis penalty if workload just changed
        let hysteresis_penalty = if self.workload_history.len() >= 2 &&
            self.workload_history[self.workload_history.len() - 1] !=
            self.workload_history[self.workload_history.len() - 2] { 0.15 } else { 0.0 };

        // Combine all factors
        let final_confidence = (base_confidence + stability_bonus - hysteresis_penalty).clamp(0.0, 1.0);

        // Round to 2 decimal places for consistency
        (final_confidence * 100.0).round() / 100.0
    }

    /// Normalize a value to a confidence score between 0 and 1
    fn normalize_confidence(&self, value: f64, min_good: f64, max_good: f64) -> f64 {
        if value >= max_good {
            1.0
        } else if value <= min_good {
            0.0
        } else {
            (value - min_good) / (max_good - min_good)
        }
    }

    /// Get information about a specific scheduler
    pub fn get_scheduler_info(&self, name: &str) -> Option<&SchedulerInfo> {
        self.scheduler_characteristics.get(name)
    }
}

/// Performance tracking for scheduler effectiveness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerPerformance {
    pub scheduler_name: String,
    pub workload_type: WorkloadType,
    pub avg_response_time: f64,
    pub avg_throughput: f64,
    pub avg_latency: f64,
    pub stability_score: f64,
    pub sample_count: u32,
    pub last_updated: u64,
}

impl SchedulerPerformance {
    pub fn new(scheduler_name: String, workload_type: WorkloadType) -> Self {
        Self {
            scheduler_name,
            workload_type,
            avg_response_time: 0.0,
            avg_throughput: 0.0,
            avg_latency: 0.0,
            stability_score: 0.0,
            sample_count: 0,
            last_updated: 0,
        }
    }

    pub fn update(&mut self, response_time: f64, throughput: f64, latency: f64, stability: f64) {
        self.avg_response_time = (self.avg_response_time * self.sample_count as f64 + response_time) / (self.sample_count + 1) as f64;
        self.avg_throughput = (self.avg_throughput * self.sample_count as f64 + throughput) / (self.sample_count + 1) as f64;
        self.avg_latency = (self.avg_latency * self.sample_count as f64 + latency) / (self.sample_count + 1) as f64;
        self.stability_score = (self.stability_score * self.sample_count as f64 + stability) / (self.sample_count + 1) as f64;
        self.sample_count += 1;
        self.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
    }

    pub fn get_overall_score(&self) -> f64 {
        // Weighted score: 40% throughput, 30% latency, 20% stability, 10% response time
        let throughput_score = (self.avg_throughput / 1000.0).min(1.0);
        let latency_score = 1.0 - (self.avg_latency / 100.0).min(1.0);
        let stability_score = self.stability_score;
        let response_score = 1.0 - (self.avg_response_time / 50.0).min(1.0);

        throughput_score * 0.4 + latency_score * 0.3 + stability_score * 0.2 + response_score * 0.1
    }
}

/// Performance feedback manager
pub struct PerformanceFeedback {
    pub performance_history: HashMap<String, SchedulerPerformance>,
    current_metrics: Option<AggregatedMetrics>,
    switch_timestamp: Option<u64>,
    stabilization_period_secs: u64,
}

impl PerformanceFeedback {
    pub fn new() -> Self {
        Self {
            performance_history: HashMap::new(),
            current_metrics: None,
            switch_timestamp: None,
            stabilization_period_secs: 30, // 30 seconds stabilization period
        }
    }

    pub fn record_switch(&mut self, scheduler_name: &str, workload_type: &WorkloadType) {
        self.switch_timestamp = Some(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        let key = format!("{}-{:?}", scheduler_name, workload_type);
        if !self.performance_history.contains_key(&key) {
            self.performance_history.insert(
                key,
                SchedulerPerformance::new(scheduler_name.to_string(), workload_type.clone()),
            );
        }
    }

    pub fn update_metrics(&mut self, metrics: &AggregatedMetrics) {
        self.current_metrics = Some(metrics.clone());

        // If we're in stabilization period after a switch, update performance
        if let Some(switch_time) = self.switch_timestamp {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            if now - switch_time >= self.stabilization_period_secs {
                self.evaluate_performance();
                self.switch_timestamp = None; // Clear after evaluation
            }
        }
    }

    fn evaluate_performance(&mut self) {
        if let Some(ref metrics) = self.current_metrics {
            // Calculate performance metrics
            let response_time = metrics.sched_avg_run_time_ns as f64 / 1_000_000.0; // Convert to ms
            let throughput = metrics.sched_timeslices_per_sec;
            let latency = metrics.cpu_iowait_percent;
            let stability = 1.0 - (metrics.cpu_max_percent - metrics.cpu_avg_percent) / 100.0;

            // Update all relevant scheduler performance records
            for (_, perf) in self.performance_history.iter_mut() {
                if perf.sample_count > 0 { // Only update existing records
                    perf.update(response_time, throughput, latency, stability);
                }
            }
        }
    }

    pub fn should_rollback(&self, current_scheduler: &str, workload_type: &WorkloadType) -> bool {
        let key = format!("{}-{:?}", current_scheduler, workload_type);

        if let Some(perf) = self.performance_history.get(&key) {
            if perf.sample_count < 3 {
                return false; // Not enough data
            }

            // Check if performance is significantly worse than baseline
            let baseline_score = self.get_baseline_score(workload_type);
            let current_score = perf.get_overall_score();

            // Rollback if performance is 20% worse than baseline
            current_score < baseline_score * 0.8
        } else {
            false
        }
    }

    fn get_baseline_score(&self, workload_type: &WorkloadType) -> f64 {
        // Get best historical performance for this workload type
        let mut best_score = 0.5; // Default baseline

        for (_, perf) in &self.performance_history {
            if &perf.workload_type == workload_type && perf.sample_count >= 3 {
                best_score = f64::max(best_score, perf.get_overall_score());
            }
        }

        best_score
    }

    pub fn get_best_scheduler(&self, workload_type: &WorkloadType) -> Option<String> {
        let mut best_scheduler = None;
        let mut best_score = 0.0;

        for (key, perf) in &self.performance_history {
            if &perf.workload_type == workload_type && perf.sample_count >= 3 {
                let score = perf.get_overall_score();
                if score > best_score {
                    best_score = score;
                    best_scheduler = Some(perf.scheduler_name.clone());
                }
            }
        }

        best_scheduler
    }
}
