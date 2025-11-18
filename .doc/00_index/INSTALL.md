---
title: Installation Guide
updated_at: 2025-11-17
owner: core-dev
lang: en
tokens_estimate: 1400
layer: 00_index
tags: [installation, setup, docker, wsl2, environment]
purpose: "Complete guide to set up the development environment"
---

# Installation Guide

Complete guide to set up the development environment.

## System Requirements

### Host System
- **OS**: Windows 11 (with WSL2) or Linux
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 20GB free space

### Software Prerequisites
- **Docker Desktop** 20.10+ (Windows/Mac) or Docker 20.10+ (Linux)
- **WSL2** with Ubuntu 22.04+ (Windows only)
- **Git**

## Quick Start (3 Steps)

### Step 1: Clone and Navigate

```bash
cd ~
git clone <your-repo-url> projects/godzilla-evan
cd projects/godzilla-evan
```

**Important**: On Windows, ensure project is in WSL2 filesystem (`/home/user/`), NOT Windows filesystem (`/mnt/c/`). See [Performance Notes](#performance-notes).

### Step 2: Start Container

```bash
docker-compose up -d
```

First build takes ~10 minutes (downloads and compiles dependencies). Subsequent starts are instant.

### Step 3: Enter Dev Environment

```bash
docker-compose exec app /bin/bash
```

You're now in the container with all development tools ready.

## Detailed Setup (Windows)

### Install WSL2

1. Open PowerShell as Administrator:
   ```powershell
   wsl --install
   ```

2. Restart computer

3. Install Ubuntu:
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```

4. Create user account when prompted

### Install Docker Desktop

1. Download Docker Desktop from https://docker.com
2. Install and restart
3. Open Docker Desktop
4. Go to Settings → General
   - Enable "Use the WSL 2 based engine"
5. Go to Settings → Resources → WSL Integration
   - Enable integration for Ubuntu-22.04
6. Apply & Restart

### Verify Installation

```bash
# Check WSL2
wsl --list --verbose

# Check Docker
docker --version
docker-compose --version

# Test Docker
docker run hello-world
```

## DNS Resolution Fix

### Problem

Docker containers may fail with:
```
dial tcp: lookup auth.docker.io on 10.255.255.254:53: i/o timeout
```

### Solution

Configure Docker to use public DNS:

1. Open Docker Desktop (Windows) or edit `/etc/docker/daemon.json` (Linux)
2. Go to Settings → Docker Engine
3. Add DNS configuration:
   ```json
   {
     "dns": ["8.8.8.8", "114.114.114.114", "1.1.1.1"]
   }
   ```
4. Apply & Restart

**Why this works**: Bypasses unreliable WSL2 DNS proxy. See [adr/003-dns.md](adr/003-dns.md) for details.

## Building the Project

### Initial Build

```bash
# Enter container
docker-compose exec app /bin/bash

# Build C++ core
cd /app/core
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Install Python package
cd /app/core/python
pip3 install -e .
```

### Verify Build

```bash
# Check kfc command
kfc --help

# Run tests (if available)
cd /app/core/build
ctest
```

## IDE Integration

### Cursor / VSCode

1. Install "Remote - WSL" extension
2. Press F1 → "WSL: Connect to WSL"
3. Open folder: `/home/huyifan/projects/godzilla-evan`
4. Edit files in IDE, build in container

### Workflow

```
Edit code (Cursor/Windows)
    ↓
Auto-sync to WSL2
    ↓
Auto-sync to container (/app)
    ↓
Build in container
```

## Container Management

### Start/Stop

```bash
docker-compose up -d      # Start in background
docker-compose ps         # Check status
docker-compose logs -f    # View logs
docker-compose down       # Stop and remove
docker-compose restart    # Restart
```

### Rebuild

```bash
docker-compose up -d --build
```

### Clean Slate

```bash
docker-compose down -v    # Stop and remove volumes
docker system prune -f    # Clean unused images
docker-compose up -d --build
```

## Troubleshooting

### Container Won't Start

**Check logs**:
```bash
docker-compose logs app
```

**Common causes**:
- Port 8080 or 9001 already in use → change ports in docker-compose.yml
- DNS issues → see [DNS Resolution Fix](#dns-resolution-fix)
- Disk space → run `docker system prune -f`

### Build Failures

**Clean and rebuild**:
```bash
docker-compose exec app /bin/bash
cd /app/core/build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Check dependencies**:
```bash
# In container
python3 --version   # Should be 3.8+
cmake --version     # Should be 3.16+
gcc --version       # Should be 9.4+
```

### Slow Performance

**Check project location**:
```bash
pwd
# Good: /home/huyifan/projects/godzilla-evan
# Bad:  /mnt/c/Users/...
```

If in `/mnt/c/`, move to WSL2 filesystem:
```bash
cd ~
cp -r /mnt/c/path/to/project projects/
cd projects/godzilla-evan
```

**Reason**: WSL2 accessing Windows filesystem is 10-50x slower. See [adr/002-wsl2.md](adr/002-wsl2.md).

### DNS Still Failing

**Test DNS**:
```bash
# From WSL2
nslookup google.com 8.8.8.8

# From container
docker exec godzilla-dev nslookup google.com
```

**Alternative fix** (WSL2 level):
```bash
# Edit /etc/wsl.conf
sudo nano /etc/wsl.conf
```

Add:
```ini
[network]
generateResolvConf = false
```

Restart WSL2:
```powershell
# In Windows PowerShell
wsl --shutdown
```

Create `/etc/resolv.conf`:
```bash
sudo rm /etc/resolv.conf
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

### "Cannot Connect to Docker Daemon"

**Check Docker is running**:
```bash
docker ps
```

**On WSL2**, Docker Desktop must be running on Windows.

**On Linux**:
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

## Performance Notes

### File System Placement

**Critical**: Project MUST be in WSL2 filesystem for good performance.

```bash
# ✓ Fast: WSL2 filesystem
/home/huyifan/projects/godzilla-evan

# ✗ Slow: Windows filesystem via WSL2
/mnt/c/Users/huyifan/projects/godzilla-evan
```

**Benchmark** (same build):
- WSL2 filesystem: 32 seconds
- Windows filesystem: 580 seconds

Reference: https://docs.microsoft.com/en-us/windows/wsl/compare-versions

### Resource Limits

Docker Desktop settings (Settings → Resources):
- **CPUs**: 4+ recommended
- **Memory**: 8GB recommended
- **Swap**: 2GB
- **Disk**: 60GB

## Deployment to Server

Same configuration works on any Linux server:

```bash
# On server
git clone <your-repo>
cd godzilla-evan

# Install Docker (if needed)
curl -fsSL https://get.docker.com | sudo sh

# Start
docker-compose up -d

# Verify
docker-compose ps
docker-compose exec app kfc --version
```

Zero modifications needed. See [adr/001-docker.md](adr/001-docker.md) for rationale.

## Environment Details

### Container Specifications

- **Base**: Ubuntu 20.04
- **Name**: godzilla-dev
- **Work Dir**: /app
- **Ports**: 8080, 9001
- **Volumes**:
  - Project code: `.:/app`
  - Build cache: `build-cache:/app/core/build`
  - Pip cache: `pip-cache:/root/.cache/pip`

### Installed Tools

- **Build**: gcc 9.4, g++, make, cmake 3.16
- **Python**: 3.8.10, pip
- **Libraries**: Boost, OpenSSL, SQLite
- **Utils**: git, vim, curl, wget

**Note**: PM2 (process manager) is NOT pre-installed. If you plan to use official test scripts (`scripts/binance_test/run.sh`), install PM2:
```bash
docker-compose exec app bash
apt-get update && apt-get install -y nodejs npm
npm install -g pm2
```

See [TESTNET.md](TESTNET.md#0-install-pm2-process-manager-️-required) for details.

### Verify Environment

```bash
docker-compose exec app /bin/bash
python3 --version
cmake --version
gcc --version
ls /app
```

## Next Steps

After successful installation:

1. Read [HACKING.md](HACKING.md) for development workflow
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
3. Try building and running a strategy

## Related Documentation

- [adr/001-docker.md](adr/001-docker.md) - Why Docker for development
- [adr/002-wsl2.md](adr/002-wsl2.md) - Why WSL2 as backend
- [adr/003-dns.md](adr/003-dns.md) - DNS resolution strategy
- [HACKING.md](HACKING.md) - Development workflow

---

Last Updated: 2025-10-28

