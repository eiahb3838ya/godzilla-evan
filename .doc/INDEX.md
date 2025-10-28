# Documentation Index

Complete documentation for godzilla-evan project.

## 🚨 Critical Reading (Start Here)

**Running Binance Testnet?** Read this FIRST to avoid common pitfalls:
1. **[TESTNET.md](TESTNET.md)** - Complete Binance setup guide with troubleshooting
   - PM2 installation (REQUIRED)
   - Account configuration (`gz_user1` naming)
   - Graceful shutdown to avoid crashes

**New to this project?** Read in order:
2. [INSTALL.md](INSTALL.md) - Docker environment setup
3. [ARCHITECTURE.md](ARCHITECTURE.md) - System design (yijinjing + wingchun)

**Hit a bug?** Check these:
4. [DEBUGGING.md](DEBUGGING.md) - Real debugging case studies with solutions
5. [LOG_LOCATIONS.md](LOG_LOCATIONS.md) - Where to find system logs

## Reference Documentation

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design (yijinjing + wingchun) |
| [HACKING.md](HACKING.md) | Development workflow and build process |
| [ORIGIN.md](ORIGIN.md) | Fork history and license info |
| [CHANGELOG.md](CHANGELOG.md) | Change history |

## Architecture Decisions (ADRs)

- [adr/001-docker.md](adr/001-docker.md) - Why Docker
- [adr/002-wsl2.md](adr/002-wsl2.md) - Why WSL2
- [adr/003-dns.md](adr/003-dns.md) - DNS strategy

## Quick Start

```bash
# Start environment
docker-compose up -d
docker-compose exec app bash

# Build C++ core
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Install Python package
cd /app/core/python && pip3 install -e .
```

For detailed setup, see [INSTALL.md](INSTALL.md).

---

Last Updated: 2025-10-24

