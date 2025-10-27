use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Workload type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum WorkloadType {
    CpuBound,
    IoBound,
    Mixed,
    LatencySensitive,
    Batch,
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
            workload_types: vec![WorkloadType::LatencySensitive, WorkloadType::Mixed],
            performance_profile: "optimized for low latency and interactive responsiveness".to_string(),
            use_cases: vec!["gaming".to_string(), "live_streaming".to_string(), "multimedia".to_string(), "real_time_audio".to_string()],
            default_args: vec!["--slice-us".to_string(), "20000".to_string()],
        });

        scheduler_characteristics.insert("scx_flash".to_string(), SchedulerInfo {
            name: "scx_flash".to_string(),
            workload_types: vec![WorkloadType::LatencySensitive, WorkloadType::Mixed],
            performance_profile: "optimized for low latency and predictable performance".to_string(),
            use_cases: vec!["multimedia".to_string(), "real_time_audio".to_string(), "predictable_performance".to_string(), "latency_sensitive".to_string()],
            default_args: vec!["--slice-us".to_string(), "4096".to_string()],
        });

        scheduler_characteristics.insert("scx_lavd".to_string(), SchedulerInfo {
            name: "scx_lavd".to_string(),
            workload_types: vec![WorkloadType::LatencySensitive, WorkloadType::Mixed],
            performance_profile: "optimized for low latency and interactivity".to_string(),
            use_cases: vec!["gaming".to_string(), "interactive_applications".to_string(), "low_latency".to_string()],
            default_args: vec!["--slice-max-us".to_string(), "5000".to_string()],
        });

        scheduler_characteristics.insert("scx_rusty".to_string(), SchedulerInfo {
            name: "scx_rusty".to_string(),
            workload_types: vec![WorkloadType::Batch, WorkloadType::Mixed],
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
        }
    }

    /// Classify the current workload based on aggregated metrics
    pub fn classify_workload(&self, metrics: &super::metrics::AggregatedMetrics) -> WorkloadType {
        // High CPU usage with low I/O wait indicates CPU-bound workload
        if metrics.cpu_avg_percent > 70.0 && metrics.cpu_iowait_percent < 5.0 {
            return WorkloadType::CpuBound;
        }

        // High I/O wait indicates I/O-bound workload
        if metrics.cpu_iowait_percent > 20.0 {
            return WorkloadType::IoBound;
        }

        // High memory usage with moderate CPU might indicate memory-bound workload
        if metrics.memory_avg_percent > 80.0 && metrics.cpu_avg_percent > 30.0 {
            return WorkloadType::Mixed;
        }

        // Low CPU usage with low I/O wait might indicate latency-sensitive workload
        if metrics.cpu_avg_percent < 30.0 && metrics.cpu_iowait_percent < 10.0 {
            return WorkloadType::LatencySensitive;
        }

        // Default to mixed for balanced workloads
        WorkloadType::Mixed
    }

    /// Recommend a scheduler based on workload classification and metrics
    pub fn recommend_scheduler(&self, metrics: &super::metrics::AggregatedMetrics) -> SchedulerRecommendation {
        let workload_type = self.classify_workload(metrics);

        // Match workload type to appropriate scheduler
        let (scheduler_name, reason) = match workload_type {
            WorkloadType::CpuBound => {
                ("scx_rusty", "High CPU utilization detected, scx_rusty provides good throughput for CPU-bound workloads")
            },
            WorkloadType::IoBound => {
                ("scx_bpfland", "High I/O wait detected, scx_bpfland prioritizes interactive tasks")
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
        match workload_type {
            WorkloadType::CpuBound => {
                // Confidence based on high CPU usage and low I/O wait
                let cpu_confidence = (metrics.cpu_avg_percent / 100.0).min(1.0);
                let io_confidence = 1.0 - (metrics.cpu_iowait_percent / 100.0).min(1.0);
                (cpu_confidence + io_confidence) / 2.0
            },
            WorkloadType::IoBound => {
                // Confidence based on high I/O wait
                (metrics.cpu_iowait_percent / 100.0).min(1.0)
            },
            WorkloadType::LatencySensitive => {
                // Confidence based on low CPU and I/O wait
                let cpu_confidence = 1.0 - (metrics.cpu_avg_percent / 100.0).min(1.0);
                let io_confidence = 1.0 - (metrics.cpu_iowait_percent / 100.0).min(1.0);
                (cpu_confidence + io_confidence) / 2.0
            },
            WorkloadType::Batch | WorkloadType::Mixed => {
                // Default confidence for other types
                0.7
            },
        }
    }

    /// Get information about a specific scheduler
    pub fn get_scheduler_info(&self, name: &str) -> Option<&SchedulerInfo> {
        self.scheduler_characteristics.get(name)
    }
}