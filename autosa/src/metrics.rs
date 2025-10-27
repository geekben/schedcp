use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// CPU statistics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuStats {
    pub timestamp: u64,
    pub user: u64,
    pub nice: u64,
    pub system: u64,
    pub idle: u64,
    pub iowait: u64,
    pub irq: u64,
    pub softirq: u64,
    pub steal: u64,
}

/// Memory statistics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub timestamp: u64,
    pub total_kb: u64,
    pub free_kb: u64,
    pub available_kb: u64,
    pub buffers_kb: u64,
    pub cached_kb: u64,
    pub used_kb: u64,
    pub used_percent: f64,
}

/// I/O statistics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoStats {
    pub timestamp: u64,
    pub device_name: String,
    pub reads_completed: u64,
    pub writes_completed: u64,
    pub sectors_read: u64,
    pub sectors_written: u64,
    pub time_reading_ms: u64,
    pub time_writing_ms: u64,
    pub io_in_progress: u64,
    pub time_in_progress_ms: u64,
}

/// Scheduler statistics from /proc/schedstat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedStats {
    pub timestamp: u64,
    pub cpu_count: usize,
    pub total_run_time: u64,
    pub total_wait_time: u64,
    pub total_timeslices: u64,
}

/// Combined system metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: u64,
    pub cpu_stats: CpuStats,
    pub memory_stats: MemoryStats,
    pub io_stats: Vec<IoStats>,
    pub sched_stats: SchedStats,
}

/// Aggregated metrics over time windows
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetrics {
    pub timestamp: u64,
    pub window_duration_secs: u64,
    pub cpu_avg_percent: f64,
    pub cpu_max_percent: f64,
    pub cpu_iowait_percent: f64,
    pub memory_avg_percent: f64,
    pub memory_max_percent: f64,
    pub io_read_bytes_per_sec: f64,
    pub io_write_bytes_per_sec: f64,
    pub io_avg_queue_depth: f64,
    pub sched_timeslices_per_sec: f64,
    pub sched_avg_run_time_ns: u64,
}

/// Metrics collector that gathers system metrics
pub struct MetricsCollector {
    samples: Vec<SystemMetrics>,
    max_samples: usize,
}

impl MetricsCollector {
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples,
        }
    }

    /// Collect all system metrics at the current time
    pub async fn collect_metrics(&self) -> Result<SystemMetrics> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("Failed to get timestamp")?
            .as_secs();

        let cpu_stats = self.collect_cpu_stats(timestamp).await?;
        let memory_stats = self.collect_memory_stats(timestamp).await?;
        let io_stats = self.collect_io_stats(timestamp).await?;
        let sched_stats = self.collect_sched_stats(timestamp).await?;

        Ok(SystemMetrics {
            timestamp,
            cpu_stats,
            memory_stats,
            io_stats,
            sched_stats,
        })
    }

    /// Collect CPU statistics from /proc/stat
    async fn collect_cpu_stats(&self, timestamp: u64) -> Result<CpuStats> {
        let stat_content = tokio::fs::read_to_string("/proc/stat")
            .await
            .context("Failed to read /proc/stat")?;

        let cpu_line = stat_content
            .lines()
            .find(|line| line.starts_with("cpu "))
            .ok_or_else(|| anyhow::anyhow!("CPU line not found in /proc/stat"))?;

        let fields: Vec<&str> = cpu_line.split_whitespace().collect();
        if fields.len() < 9 {
            anyhow::bail!("Invalid CPU line format");
        }

        Ok(CpuStats {
            timestamp,
            user: fields[1].parse().unwrap_or(0),
            nice: fields[2].parse().unwrap_or(0),
            system: fields[3].parse().unwrap_or(0),
            idle: fields[4].parse().unwrap_or(0),
            iowait: fields[5].parse().unwrap_or(0),
            irq: fields[6].parse().unwrap_or(0),
            softirq: fields[7].parse().unwrap_or(0),
            steal: fields[8].parse().unwrap_or(0),
        })
    }

    /// Collect memory statistics from /proc/meminfo
    async fn collect_memory_stats(&self, timestamp: u64) -> Result<MemoryStats> {
        let meminfo_content = tokio::fs::read_to_string("/proc/meminfo")
            .await
            .context("Failed to read /proc/meminfo")?;

        let mut stats = HashMap::new();
        for line in meminfo_content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                let key = parts[0].trim_end_matches(':');
                if let Ok(value) = parts[1].parse::<u64>() {
                    stats.insert(key.to_string(), value);
                }
            }
        }

        let total = stats.get("MemTotal").copied().unwrap_or(0);
        let free = stats.get("MemFree").copied().unwrap_or(0);
        let available = stats.get("MemAvailable").copied().unwrap_or(0);
        let buffers = stats.get("Buffers").copied().unwrap_or(0);
        let cached = stats.get("Cached").copied().unwrap_or(0);
        let used = total.saturating_sub(free).saturating_sub(buffers).saturating_sub(cached);
        let used_percent = if total > 0 {
            (used as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Ok(MemoryStats {
            timestamp,
            total_kb: total,
            free_kb: free,
            available_kb: available,
            buffers_kb: buffers,
            cached_kb: cached,
            used_kb: used,
            used_percent,
        })
    }

    /// Collect I/O statistics from /proc/diskstats
    async fn collect_io_stats(&self, timestamp: u64) -> Result<Vec<IoStats>> {
        // Use procfs crate to read diskstats
        let disk_stats = procfs::diskstats()
            .context("Failed to read diskstats")?;

        let mut io_stats = Vec::new();
        for disk_stat in disk_stats {
            // Skip loop devices and ram disks
            if disk_stat.name.starts_with("loop") || disk_stat.name.starts_with("ram") {
                continue;
            }

            io_stats.push(IoStats {
                timestamp,
                device_name: disk_stat.name,
                reads_completed: disk_stat.reads,
                writes_completed: disk_stat.writes,
                sectors_read: disk_stat.sectors_read,
                sectors_written: disk_stat.sectors_written,
                time_reading_ms: disk_stat.time_reading,
                time_writing_ms: disk_stat.time_writing,
                io_in_progress: disk_stat.in_progress,
                time_in_progress_ms: disk_stat.time_in_progress,
            });
        }

        Ok(io_stats)
    }

    /// Collect scheduler statistics from /proc/schedstat
    async fn collect_sched_stats(&self, timestamp: u64) -> Result<SchedStats> {
        let schedstat_content = tokio::fs::read_to_string("/proc/schedstat")
            .await
            .context("Failed to read /proc/schedstat")?;

        let mut cpu_count = 0;
        let mut total_run_time = 0u64;
        let mut total_wait_time = 0u64;
        let mut total_timeslices = 0u64;

        for line in schedstat_content.lines() {
            if line.starts_with("cpu") {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() >= 9 {
                    cpu_count += 1;
                    total_run_time += fields[7].parse().unwrap_or(0);
                    total_wait_time += fields[8].parse().unwrap_or(0);
                }
            } else if line.starts_with("domain") {
                // Skip domain lines for now
                continue;
            } else if !line.starts_with("version") && !line.starts_with("timestamp") {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() >= 1 {
                    total_timeslices += fields[0].parse().unwrap_or(0);
                }
            }
        }

        Ok(SchedStats {
            timestamp,
            cpu_count,
            total_run_time,
            total_wait_time,
            total_timeslices,
        })
    }

    /// Add a metrics sample to our collection
    pub fn add_sample(&mut self, metrics: SystemMetrics) {
        self.samples.push(metrics);
        // Keep only the most recent samples
        if self.samples.len() > self.max_samples {
            self.samples.drain(0..(self.samples.len() - self.max_samples));
        }
    }

    /// Calculate aggregated metrics over a time window
    pub fn calculate_aggregated_metrics(&self, window_duration_secs: u64) -> Option<AggregatedMetrics> {
        if self.samples.len() < 2 {
            return None;
        }

        // Find samples within the time window
        let latest_timestamp = self.samples.last().unwrap().timestamp;
        let window_start = latest_timestamp.saturating_sub(window_duration_secs);

        let samples_in_window: Vec<&SystemMetrics> = self.samples
            .iter()
            .filter(|sample| sample.timestamp >= window_start)
            .collect();

        if samples_in_window.len() < 2 {
            return None;
        }

        // Calculate CPU usage percentages
        let mut cpu_percentages = Vec::new();
        let mut cpu_iowait_percentages = Vec::new();
        for i in 1..samples_in_window.len() {
            let prev = &samples_in_window[i - 1].cpu_stats;
            let curr = &samples_in_window[i].cpu_stats;

            let prev_total = prev.user + prev.nice + prev.system + prev.idle
                + prev.iowait + prev.irq + prev.softirq + prev.steal;
            let curr_total = curr.user + curr.nice + curr.system + curr.idle
                + curr.iowait + curr.irq + curr.softirq + curr.steal;

            let total_diff = curr_total.saturating_sub(prev_total);
            let idle_diff = curr.idle.saturating_sub(prev.idle);
            let iowait_diff = curr.iowait.saturating_sub(prev.iowait);

            if total_diff > 0 {
                let usage_percent = ((total_diff - idle_diff) as f64 / total_diff as f64) * 100.0;
                let iowait_percent = (iowait_diff as f64 / total_diff as f64) * 100.0;
                cpu_percentages.push(usage_percent);
                cpu_iowait_percentages.push(iowait_percent);
            }
        }

        let cpu_avg_percent = if !cpu_percentages.is_empty() {
            cpu_percentages.iter().sum::<f64>() / cpu_percentages.len() as f64
        } else {
            0.0
        };

        let cpu_max_percent = cpu_percentages
            .iter()
            .copied()
            .fold(0.0f64, f64::max);

        let cpu_iowait_percent = if !cpu_iowait_percentages.is_empty() {
            cpu_iowait_percentages.iter().sum::<f64>() / cpu_iowait_percentages.len() as f64
        } else {
            0.0
        };

        // Memory statistics
        let memory_avg_percent = if !samples_in_window.is_empty() {
            samples_in_window.iter()
                .map(|s| s.memory_stats.used_percent)
                .sum::<f64>() / samples_in_window.len() as f64
        } else {
            0.0
        };

        let memory_max_percent = samples_in_window.iter()
            .map(|s| s.memory_stats.used_percent)
            .fold(0.0f64, f64::max);

        // I/O statistics (calculate rates per second)
        let mut io_read_bytes_total = 0u64;
        let mut io_write_bytes_total = 0u64;
        let mut io_time_in_progress_total = 0u64;
        let mut io_sample_count = 0u64;

        for sample in &samples_in_window {
            for io_stat in &sample.io_stats {
                io_read_bytes_total += io_stat.sectors_read * 512; // Convert sectors to bytes
                io_write_bytes_total += io_stat.sectors_written * 512;
                io_time_in_progress_total += io_stat.time_in_progress_ms;
                io_sample_count += 1;
            }
        }

        let window_duration = (samples_in_window.last().unwrap().timestamp - samples_in_window.first().unwrap().timestamp) as f64;
        let io_read_bytes_per_sec = if window_duration > 0.0 {
            io_read_bytes_total as f64 / window_duration
        } else {
            0.0
        };

        let io_write_bytes_per_sec = if window_duration > 0.0 {
            io_write_bytes_total as f64 / window_duration
        } else {
            0.0
        };

        let io_avg_queue_depth = if io_sample_count > 0 {
            io_time_in_progress_total as f64 / io_sample_count as f64 / 1000.0 // Convert to average queue depth
        } else {
            0.0
        };

        // Scheduler statistics
        let sched_timeslices_per_sec = if window_duration > 0.0 && !samples_in_window.is_empty() {
            let first = samples_in_window.first().unwrap().sched_stats.total_timeslices;
            let last = samples_in_window.last().unwrap().sched_stats.total_timeslices;
            (last - first) as f64 / window_duration
        } else {
            0.0
        };

        let sched_avg_run_time_ns = if !samples_in_window.is_empty() {
            samples_in_window.iter()
                .map(|s| s.sched_stats.total_run_time)
                .sum::<u64>() / samples_in_window.len() as u64
        } else {
            0
        };

        Some(AggregatedMetrics {
            timestamp: latest_timestamp,
            window_duration_secs,
            cpu_avg_percent,
            cpu_max_percent,
            cpu_iowait_percent,
            memory_avg_percent,
            memory_max_percent,
            io_read_bytes_per_sec,
            io_write_bytes_per_sec,
            io_avg_queue_depth,
            sched_timeslices_per_sec,
            sched_avg_run_time_ns,
        })
    }
}