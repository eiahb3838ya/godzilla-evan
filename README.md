# Godzilla-Evan

Low-latency cryptocurrency trading framework with event-sourced architecture.

## Features

- **Sub-millisecond latency**: C++ core with journal-based event system
- **Multi-exchange**: Binance (spot/futures), extensible to other exchanges
- **Python strategies**: Write trading logic in Python with full C++ performance
- **Event sourcing**: Complete audit trail with time-travel debugging
- **Production-ready**: Docker deployment, PM2 process management

## Quick Start

```bash
# Clone and start
git clone <your-repo>
cd godzilla-evan
docker-compose up -d

# Enter dev environment
docker-compose exec app /bin/bash

# Run a strategy
python3 core/python/dev_run.py -l info strategy \
  -n demo_spot \
  -p strategies/demo_spot.py
```

## Documentation

All documentation is in [`.doc/`](.doc/):

- **[START.md](.doc/START.md)** - AI assistant onboarding
- **[DESIGN.md](.doc/00_index/DESIGN.md)** - System architecture
- **[Strategy Guide](.doc/10_modules/strategy_framework.md)** - Write trading strategies
- **[API Reference](.doc/30_contracts/strategy_context_api.md)** - Complete API docs
- **[Operations](.doc/90_operations/pm2_startup_guide.md)** - Deployment & operations

**For AI assistants**: Run `follow .doc/START.md` to load project context.

## Architecture

```
┌─────────────────────────────────────────┐
│  Python Strategy Layer                  │
│  (User trading logic)                   │
└──────────────┬──────────────────────────┘
               │ pybind11
┌──────────────▼──────────────────────────┐
│  Wingchun (C++)                         │
│  • Strategy runtime                     │
│  • Order management                     │
│  • Position tracking                    │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Yijinjing (C++)                        │
│  • Event sourcing (Journal)             │
│  • Message passing (~50μs)              │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Exchange Gateways                      │
│  • Binance REST/WebSocket               │
│  • Order routing                        │
└─────────────────────────────────────────┘
```

## Project Structure

```
.
├── core/
│   ├── cpp/              # C++ core (yijinjing, wingchun)
│   ├── python/           # Python bindings
│   └── extensions/       # Exchange connectors
├── strategies/           # Trading strategies
├── .doc/                 # Documentation
│   ├── 10_modules/       # Module guides
│   ├── 30_contracts/     # API contracts
│   └── 90_operations/    # Operations guides
└── docker-compose.yml
```

## Development

See [.doc/00_index/HACKING.md](.doc/00_index/HACKING.md) for:
- Build instructions
- Development workflow
- Testing strategies

## License

Apache 2.0

## Credits

Modified fork of [kungfu](https://github.com/kungfu-trader/kungfu) by Keren Dong.
