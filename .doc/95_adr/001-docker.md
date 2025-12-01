# ADR-001: Docker-based Development Environment

Date: 2025-10-22 (Documented retroactively)

## Status

Accepted (Implemented)

## Context

The project (kungfu fork) requires a consistent development environment across different platforms. Development team uses heterogeneous systems (Windows, macOS, Linux), but production runs on Linux.

### Problems

1. **Platform heterogeneity**: Team members use different operating systems
2. **Complex dependencies**: kungfu requires Boost, nanomsg, SQLite, CMake, GCC, Python
3. **Windows challenges**: Native Windows builds face toolchain complexity
4. **Environment drift**: "Works on my machine" problems
5. **Production parity**: Need to match production Linux environment

### Constraints

- Must support Windows developers (primary platform)
- Must match production Linux environment
- Must be easy to onboard new developers
- Build times should be reasonable
- No requirement for dedicated hardware

## Decision

Use **Docker containers** for the development environment.

### Implementation

1. **Base image**: Ubuntu 20.04 (matches production)
2. **Orchestration**: docker-compose for service management
3. **Volume mounting**: Bind mount project directory for live code editing
4. **Tool installation**: All build tools and dependencies in container
5. **Entry point**: `kfc` CLI available in container

### Configuration

```yaml
# docker-compose.yml
services:
  app:
    build:
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app  # Live code sync
    ports:
      - "8080:8080"
      - "9001:9001"
```

### Workflow

1. Developer edits code on host (Windows/Mac) using Cursor/VSCode
2. Changes automatically sync to container via volume mount
3. Build and run commands execute inside container
4. Ports exposed for browser/API access from host

## Consequences

### Positive

- Environment consistency across all developers
- Easy onboarding: `docker-compose up -d` and start coding
- Production parity: dev environment matches production Linux
- Isolation: no system-wide dependencies or conflicts
- Portability: can move project to any machine with Docker
- Clean slate: easy to reset with rebuild

### Negative

- DNS resolution issues in WSL2 (see ADR-003)
- File I/O overhead on macOS/Windows volume mounts
- Requires basic Docker knowledge
- Disk space: container images ~2-3GB
- WSL2 required for Windows (not WSL1)

### Neutral

- First build is slow (~10 min), subsequent builds are fast
- Requires IDE integration (WSL extension for Cursor/VSCode)
- Debugging requires attaching to container

## Alternatives Considered

### Alternative 1: Native Windows Build

Use MSVC or MinGW to build natively on Windows.

**Rejected because**:
- Toolchain complexity (Boost compilation is painful on Windows)
- Windows-specific bugs different from production Linux
- Significant time investment for non-production platform
- kungfu upstream primarily targets Linux

### Alternative 2: Virtual Machine

Use VMware or VirtualBox to run full Linux VM.

**Rejected because**:
- Higher resource overhead (4GB+ RAM)
- Slower startup time
- More complex networking setup
- Harder to share files between host and VM

### Alternative 3: Dual Boot Linux

Install Linux alongside Windows.

**Rejected because**:
- Requires disk partitioning
- Cannot use Windows tools simultaneously
- Inconvenient (requires reboot to switch)
- High barrier for team members unfamiliar with Linux

### Alternative 4: GitHub Codespaces

Use cloud-based development environment.

**Rejected because**:
- Requires internet connection
- Latency for code editing
- Monthly cost per developer
- Data security concerns (proprietary trading strategies)

## Related Decisions

- ADR-002: Use WSL2 as Docker backend on Windows
- ADR-003: DNS resolution strategy for Docker in WSL2

## Notes

### Performance

On WSL2 + Docker Desktop:
- First build: ~8-10 minutes
- Incremental rebuild: ~30 seconds
- File sync latency: <100ms (acceptable)

### Critical: Project Placement

**Must** place project in WSL2 filesystem (`~/projects/`), NOT Windows filesystem (`/mnt/c/`).

- Windows filesystem via WSL2: 10-50x slower
- WSL2 native filesystem: near-native Linux performance

Reference: https://docs.microsoft.com/en-us/windows/wsl/compare-versions

### Future Improvements

1. Multi-stage builds: separate build and runtime images
2. Build caching: ccache for C++ compilation
3. Devcontainer: devcontainer.json for seamless IDE integration
4. CI/CD: use same Docker image in pipeline

---

Last Updated: 2025-10-22

