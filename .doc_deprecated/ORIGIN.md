# Project Origin

This document explains the origin and history of this project.

## Identity

**Code Name**: kungfu  
**Repository Name**: godzilla-evan  
**Type**: Modified fork  
**License**: Apache 2.0 (maintained from upstream)

## Upstream

**Original Project**: kungfu (功夫) trading framework  
**Original Author**: Keren Dong  
**Presumed Source**: https://github.com/kungfu-origin/kungfu  
**License**: Apache 2.0

## Fork Details

**Forked By**: godzilla.dev team  
**Modifier**: kx@godzilla.dev  
**Fork Date**: ~March 2025  
**Upstream Tracking**: Not syncing with upstream

## Why Fork?

[To be documented based on team's actual rationale]

Possible reasons:
- Custom exchange integrations
- Proprietary trading strategies
- Different performance requirements
- Organization-specific features
- Modifications not suitable for upstream

## Core Components (from upstream)

The fundamental architecture from kungfu is preserved:

### yijinjing (易筋經)
Journal-based event system:
- High-performance message queue
- Time-series data storage
- Event sourcing pattern

### wingchun (詠春)
Trading gateway abstraction:
- Order management system
- Position and risk tracking
- Multi-exchange support

### Infrastructure
- CMake build system (C++17)
- Python bindings via pybind11
- CLI tool: `kfc` (kungfu cli)

## Modifications

[To be documented as changes are made]

Known modifications as of 2025-10-22:
- Project renamed to "godzilla-evan" in some places
- Custom Docker development environment
- Enhanced documentation structure
- [TODO: Document actual code changes from upstream]

## Code Attribution

Original code comments preserve attribution:

```
This is source code modified under the Apache License 2.0.
Original Author: Keren Dong
Modifier: kx@godzilla.dev
Modification date: March 3, 2025
```

All modifications maintain Apache 2.0 license compatibility.

## Upstream Relationship

**Status**: Independent fork, not tracking upstream

This fork:
- Does NOT regularly sync with kungfu upstream
- Does NOT automatically merge upstream updates
- May diverge significantly over time

To check upstream for updates:
1. Visit kungfu repository
2. Compare commits since fork point
3. Manually evaluate and port desired changes
4. Test thoroughly before merging
5. Document in CHANGELOG

## For Contributors

When modifying code:

1. Preserve original attribution in source files
2. Add your modification date and identifier
3. Document changes in CHANGELOG
4. Maintain Apache 2.0 license compatibility
5. Test changes don't break existing functionality

When contributing back to kungfu upstream:
- Verify changes are Apache 2.0 compatible
- Check if upstream wants the contribution
- Follow kungfu's contribution guidelines
- Remove godzilla-specific modifications

## License Compliance

This project maintains the Apache 2.0 license from upstream.

Key requirements:
- ✓ Original license included (see LICENSE file)
- ✓ Original attribution preserved in source files
- ✓ Modifications clearly marked
- ✓ Same license applied to derivative work

## References

- Apache 2.0 License: https://www.apache.org/licenses/LICENSE-2.0
- godzilla.dev: https://godzilla.dev
- Original kungfu: [link when identified]

## Questions

For questions about:
- **Original kungfu design**: Contact kungfu maintainers
- **godzilla-evan modifications**: Contact kx@godzilla.dev
- **License compliance**: See LICENSE file

---

Last Updated: 2025-10-22

