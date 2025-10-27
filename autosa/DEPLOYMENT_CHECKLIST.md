# AutoSA Deployment Checklist

## Pre-Deployment

### 1. Verify Target System Requirements
- [ ] Linux kernel 6.12+ with sched-ext support
- [ ] Compatible CPU architecture (x86_64, ARM64, etc.)
- [ ] Sufficient disk space (binary is ~2MB)
- [ ] Root privileges available

### 2. Prepare Deployment Environment
- [ ] Build binary for target architecture if needed
- [ ] Test binary on similar system
- [ ] Prepare configuration files if needed
- [ ] Document deployment procedure

## Deployment Steps

### Option A: Simple Deployment
1. [ ] Copy binary to target system:
   ```bash
   scp target/release/autosa user@target:/tmp/
   ```
2. [ ] Install binary on target system:
   ```bash
   sudo cp /tmp/autosa /usr/local/bin/
   sudo chmod +x /usr/local/bin/autosa
   ```
3. [ ] Test basic functionality:
   ```bash
   autosa --help
   ```

### Option B: Production Deployment
1. [ ] Copy binary to target system
2. [ ] Install binary:
   ```bash
   sudo cp /tmp/autosa /usr/local/bin/
   sudo chmod +x /usr/local/bin/autosa
   ```
3. [ ] Create systemd service:
   ```bash
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
   ```
4. [ ] Enable and start service:
   ```bash
   sudo systemctl enable autosa
   sudo systemctl start autosa
   ```
5. [ ] Verify service status:
   ```bash
   sudo systemctl status autosa
   ```

## Post-Deployment Verification

### 1. Basic Functionality Tests
- [ ] Verify binary executes: `autosa --help`
- [ ] Test short run: `timeout 5s autosa start`
- [ ] Check process is running: `ps aux | grep autosa`

### 2. System Integration Tests
- [ ] Verify metrics collection works
- [ ] Check workload classification accuracy
- [ ] Confirm scheduler recommendations are appropriate
- [ ] Monitor system resource usage

### 3. Production Environment Tests
- [ ] Verify systemd service management
- [ ] Check automatic restart on failure
- [ ] Monitor logs: `journalctl -u autosa`
- [ ] Test configuration persistence

## Monitoring and Maintenance

### 1. Ongoing Monitoring
- [ ] Regular log review
- [ ] Performance impact assessment
- [ ] Scheduler switching effectiveness
- [ ] Resource utilization tracking

### 2. Updates and Maintenance
- [ ] Backup configuration before updates
- [ ] Test new versions in staging first
- [ ] Plan for scheduled maintenance windows
- [ ] Monitor for kernel updates that may affect sched-ext

### 3. Troubleshooting
- [ ] Check system logs: `journalctl -u autosa`
- [ ] Verify kernel sched-ext support
- [ ] Confirm root privileges
- [ ] Validate configuration parameters

## Rollback Procedure

If issues occur:
1. [ ] Stop the service: `sudo systemctl stop autosa`
2. [ ] Revert to previous version if available
3. [ ] Restore previous configuration
4. [ ] Restart service or troubleshoot further

## Security Considerations

- [ ] Limit service privileges to minimum required
- [ ] Regular security audits
- [ ] Monitor for unauthorized access
- [ ] Keep system updated with security patches