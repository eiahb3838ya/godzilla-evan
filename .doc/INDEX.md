# Documentation Index

Complete documentation for godzilla-evan project.

## Essential Reading

**New to this project?** Start here:

1. [ORIGIN.md](ORIGIN.md) - Understand the project's history (kungfu fork)
2. [INSTALL.md](INSTALL.md) - Set up your development environment
3. [HACKING.md](HACKING.md) - Learn the development workflow

## Core Documentation

### ORIGIN.md
Explains that this is a fork of the kungfu trading framework.
- Project identity and history
- Fork relationship
- License information
- Code attribution

### INSTALL.md
Complete environment setup guide.
- System requirements (Docker, WSL2)
- Quick start (3 steps)
- DNS troubleshooting
- Common issues and solutions

### HACKING.md
Development workflow and code structure.
- Code organization (yijinjing, wingchun)
- Build process
- Testing
- Contribution guidelines

### ARCHITECTURE.md
System architecture and design.
- Overall architecture
- yijinjing event system
- wingchun trading gateway
- Data flow and performance

### CHANGELOG.md
Structured change history.
- Version history
- Major changes
- Migration notes

## Architecture Decision Records

Important decisions and their rationale:

- **[adr/001-docker.md](adr/001-docker.md)** - Why Docker for development
- **[adr/002-wsl2.md](adr/002-wsl2.md)** - Why WSL2 as Docker backend
- **[adr/003-dns.md](adr/003-dns.md)** - DNS resolution strategy

## Module Documentation

Detailed documentation for core modules (when available):

- `modules/yijinjing.md` - Event system internals
- `modules/wingchun.md` - Trading gateway architecture

## Scripts

Useful scripts in `.doc/scripts/`:

- `verify-commands.sh` - Verify all documentation commands work

## Quick Reference

### Quick Start
```bash
cd /home/huyifan/projects/godzilla-evan
docker-compose up -d
docker-compose exec app /bin/bash
```

### Build
```bash
# In container
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Container Management
```bash
docker-compose ps                 # Status
docker-compose logs -f app        # Logs
docker-compose restart            # Restart
docker-compose down               # Stop
```

## Context Engineering

For AI/LLM context management:

- **.context/DESIGN.md** - Context management design principles
- **.context/index.yaml** - Document metadata and dependencies
- **.context/modules.yaml** - Context loading strategies

## Getting Help

1. Check [INSTALL.md](INSTALL.md) troubleshooting section
2. Review relevant ADR for design rationale
3. Run diagnostic script: `.doc/scripts/diagnostic.sh`

## Contributing

See [HACKING.md](HACKING.md) for development workflow and coding standards.

---

Last Updated: 2025-10-22

