# Godzilla-Evan

A low-latency automated crypto trading framework based on kungfu.

## What is This?

This project is a modified fork of the kungfu trading framework, customized for godzilla.dev's use case.

**Core Components**:
- **yijinjing** (易筋經): Journal-based event system for high-frequency trading
- **wingchun** (詠春): Trading gateway abstraction with order management
- **C++/Python**: Low-latency C++ core with Python bindings

**License**: Apache 2.0 (maintained from upstream)

For project origin and fork details, see [.doc/ORIGIN.md](.doc/ORIGIN.md)

## Quick Start

Three steps to start development:

```bash
# 1. Navigate to project
cd /home/huyifan/projects/godzilla-evan

# 2. Start container
docker-compose up -d

# 3. Enter dev environment
docker-compose exec app /bin/bash
```

For detailed setup instructions, see [.doc/INSTALL.md](.doc/INSTALL.md)

## Documentation

Complete documentation is in the `.doc/` directory:

- **[.doc/INDEX.md](.doc/INDEX.md)** - Documentation index
- **[.doc/INSTALL.md](.doc/INSTALL.md)** - Environment setup guide
- **[.doc/HACKING.md](.doc/HACKING.md)** - Development workflow
- **[.doc/ARCHITECTURE.md](.doc/ARCHITECTURE.md)** - System architecture

## Project Structure

```
godzilla-evan/
├── core/                # Core code (C++/Python)
│   ├── cpp/
│   │   ├── yijinjing/  # Event system
│   │   └── wingchun/   # Trading gateway
│   └── python/
├── strategies/          # Trading strategies
├── .doc/               # Documentation
└── docker-compose.yml  # Dev environment
```

## Development

Edit code on host (Windows/Mac), build in container:

```bash
# In container
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

See [.doc/HACKING.md](.doc/HACKING.md) for development workflow.

## Deployment

Same Docker configuration works on any Linux server:

```bash
git clone <your-repo>
cd godzilla-evan
docker-compose up -d
```

Zero modifications needed.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details