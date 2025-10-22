# Development Guide

How to develop and contribute to this project.

## Code Structure

```
godzilla-evan/
├── core/
│   ├── cpp/                    # C++ core
│   │   ├── yijinjing/         # Event system (易筋經)
│   │   │   ├── journal/       # Journal storage
│   │   │   ├── time/          # Time management
│   │   │   └── io/            # I/O operations
│   │   └── wingchun/          # Trading gateway (詠春)
│   │       ├── gateway/       # Exchange connectors
│   │       ├── strategy/      # Strategy engine
│   │       └── broker/        # Order routing
│   ├── python/                # Python bindings
│   │   └── kungfu/            # Main package
│   │       ├── command/       # CLI commands
│   │       ├── yijinjing/     # Python API for yijinjing
│   │       └── wingchun/      # Python API for wingchun
│   ├── deps/                  # Third-party dependencies
│   └── build/                 # Build output (in container)
├── strategies/                # Trading strategies
│   ├── helloworld/
│   └── triangular_arbitrage/
└── scripts/                   # Utility scripts
```

## Core Modules

### yijinjing (易筋經)

Journal-based event system for high-frequency trading.

**Purpose**: Persistent event storage and replay  
**Language**: C++17  
**Key Concepts**:
- Journal: Append-only event log
- Reader: Read events from journal
- Writer: Write events to journal
- Frame: Time-indexed event container

**Location**: `core/cpp/yijinjing/`

### wingchun (詠春)

Trading gateway abstraction layer.

**Purpose**: Unified interface for multiple exchanges  
**Language**: C++17 + Python  
**Key Concepts**:
- Gateway: Exchange connector
- Strategy: Trading logic
- Broker: Order management
- Position: Position tracking

**Location**: `core/cpp/wingchun/`

## Development Workflow

### 1. Set Up Environment

See [INSTALL.md](INSTALL.md) for initial setup.

### 2. Edit Code

**On Host** (Windows/Mac/Linux):
- Use Cursor/VSCode with WSL extension
- Edit files in `/home/user/projects/godzilla-evan`
- Changes auto-sync to container

### 3. Build

**In Container**:
```bash
docker-compose exec app /bin/bash
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Build Types**:
- `Release`: Optimized (-O3)
- `Debug`: Debug symbols (-g)
- `RelWithDebInfo`: Optimized + debug symbols

### 4. Test

```bash
# Unit tests (if available)
cd /app/core/build
ctest

# Integration test - run a strategy
kfc strategy run --path /app/strategies/helloworld
```

### 5. Debug

**GDB**:
```bash
# Build with debug symbols
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)

# Run under GDB
gdb --args ./your_binary
```

**Logging**:
```cpp
// In C++ code
SPDLOG_INFO("Message: {}", value);
SPDLOG_ERROR("Error: {}", error);
```

**Python**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Build System

### CMake Structure

```
core/
├── CMakeLists.txt           # Root CMake
├── cpp/
│   ├── CMakeLists.txt       # C++ projects
│   ├── yijinjing/
│   │   └── CMakeLists.txt   # yijinjing lib
│   └── wingchun/
│       └── CMakeLists.txt   # wingchun lib
└── python/
    └── setup.py             # Python package
```

### Build Options

```bash
# Release build (default)
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Verbose build
make VERBOSE=1
```

### Clean Build

```bash
cd /app/core/build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Python Development

### Package Structure

```python
kungfu/
├── __init__.py
├── command/          # CLI commands
│   ├── __init__.py
│   └── strategy.py
├── yijinjing/        # Python bindings
└── wingchun/         # Python bindings
```

### Install in Development Mode

```bash
cd /app/core/python
pip3 install -e .
```

Changes to Python files take effect immediately.

### CLI Commands

```bash
# Main command
kfc --help

# Run strategy
kfc strategy run --path /app/strategies/helloworld

# List strategies
kfc strategy list
```

## Testing

### Unit Tests

C++ tests use GoogleTest (if implemented):

```bash
cd /app/core/build
ctest --output-on-failure
```

### Strategy Testing

```bash
# Test helloworld strategy
kfc strategy run --path /app/strategies/helloworld

# Check logs
tail -f /app/runtime/log/*.log
```

### Manual Testing

```bash
# Start service
kfc service start

# Check status
kfc service status

# Stop service
kfc service stop
```

## Coding Standards

### C++ Style

- **Standard**: C++17
- **Formatting**: Follow existing code style
- **Naming**:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`
  - Constants: `UPPER_CASE`

Example:
```cpp
class JournalWriter {
public:
    void write_frame(const Frame& frame);
private:
    int64_t frame_count_;
};
```

### Python Style

- **Standard**: PEP 8
- **Formatting**: 4 spaces (no tabs)
- **Naming**:
  - Classes: `PascalCase`
  - Functions: `snake_case`
  - Variables: `snake_case`

Example:
```python
class StrategyContext:
    def on_order(self, order):
        self.log_order(order)
```

### Comments

```cpp
// Single-line comment

/**
 * Multi-line documentation
 * @param param_name Parameter description
 * @return Return value description
 */
```

## Git Workflow

### Commit Messages

Format:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Tests
- `chore`: Build/tooling

Example:
```
feat: add triangular arbitrage strategy

Implement triangular arbitrage detection across three currency pairs.
Uses yijinjing for event sourcing and wingchun for order execution.

Closes #123
```

### Branching

```bash
# Create feature branch
git checkout -b feature/your-feature

# Make changes
git add .
git commit -m "feat: your feature"

# Push
git push origin feature/your-feature
```

## Common Tasks

### Add a New Strategy

1. Create directory in `strategies/`:
   ```bash
   mkdir strategies/my_strategy
   ```

2. Create strategy file:
   ```python
   # strategies/my_strategy/strategy.py
   from kungfu.wingchun.strategy import Strategy
   
   class MyStrategy(Strategy):
       def on_quote(self, quote):
           # Your logic here
           pass
   ```

3. Run strategy:
   ```bash
   kfc strategy run --path /app/strategies/my_strategy
   ```

### Add a New Exchange Gateway

1. Create gateway file in `core/cpp/wingchun/gateway/`
2. Implement `Gateway` interface
3. Register gateway in CMakeLists.txt
4. Build and test

### Modify yijinjing Core

1. Edit files in `core/cpp/yijinjing/`
2. Rebuild:
   ```bash
   cd /app/core/build
   make -j$(nproc)
   ```
3. Test changes
4. Update tests if needed

## Performance Profiling

### CPU Profiling

Use Remotery (included in deps):

```cpp
#include "Remotery.h"

// In code
rmt_ScopedCPUSample(FunctionName, 0);
```

### Memory Profiling

Use Valgrind:

```bash
# Install in container
apt-get install valgrind

# Run
valgrind --leak-check=full ./your_binary
```

## Debugging Tips

### Container is Running But Can't Connect

```bash
docker-compose ps
docker-compose logs app
```

### Build Fails with Missing Dependencies

```bash
# Check if dependencies exist
ls /app/core/deps

# Rebuild container
docker-compose up -d --build
```

### Python Import Errors

```bash
# Check Python path
echo $PYTHONPATH

# Reinstall package
cd /app/core/python
pip3 install -e .
```

## Related Documentation

- [INSTALL.md](INSTALL.md) - Environment setup
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [adr/001-docker.md](adr/001-docker.md) - Docker rationale
- [adr/002-wsl2.md](adr/002-wsl2.md) - WSL2 rationale

## Getting Help

1. Check [INSTALL.md](INSTALL.md) troubleshooting
2. Review relevant code in `core/cpp/`
3. Check logs in `/app/runtime/log/`
4. Search for similar code patterns

---

Last Updated: 2025-10-22

