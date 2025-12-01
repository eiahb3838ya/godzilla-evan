# Binance Market Toggle Tests (ADR-004)

Tests for the market toggle feature that allows disabling Spot or Futures markets via configuration.

## Test Files

1. **test_market_toggle.cpp** - Unit tests for configuration parsing
2. **integration_test_market_toggle.sh** - Integration tests for database configuration

## Running Tests

### Unit Tests (C++)

```bash
# From project root
cd /home/huyifan/projects/godzilla-evan/core/build

# Build tests (if not already built)
cmake .. && make

# Run specific test
./test_market_toggle

# Or run all tests
ctest -R market_toggle
```

Expected output:
```
[==========] Running 7 tests from 1 test suite.
[----------] Global test environment set-up.
[----------] 7 tests from BinanceMarketToggle
[ RUN      ] BinanceMarketToggle.DefaultBothEnabled
[       OK ] BinanceMarketToggle.DefaultBothEnabled (0 ms)
[ RUN      ] BinanceMarketToggle.DisableSpotOnly
[       OK ] BinanceMarketToggle.DisableSpotOnly (0 ms)
...
[==========] 7 tests from 1 test suite ran. (X ms total)
[  PASSED  ] 7 tests.
```

### Integration Tests (Shell)

```bash
cd /home/huyifan/projects/godzilla-evan/core/extensions/binance/test

# Run integration test
./integration_test_market_toggle.sh
```

Expected output:
```
=== Binance Market Toggle Integration Test ===

[TEST] Test 1: Backward compatibility - default both enabled
[PASS] Config stored successfully (backward compatible)
[TEST] Test 2: Disable Spot market only
[PASS] Spot disabled successfully
...
[TEST] Test 5: Verify config can be read back
✓ Config structure valid
[PASS] Config read-back successful

=== Test Summary ===
All integration tests passed!
```

## Test Coverage

### Unit Tests Cover:

1. **Backward Compatibility**: Config without flags defaults to both enabled
2. **Disable Spot Only**: Spot disabled, Futures defaults to enabled
3. **Disable Futures Only**: Futures disabled, Spot defaults to enabled
4. **Explicit Enable Both**: Both explicitly set to true
5. **Disable Both**: Both explicitly set to false
6. **URL Defaults Preserved**: Hardcoded URLs still set correctly
7. **Extra Fields Ignored**: Unknown JSON fields don't break parsing

### Integration Tests Cover:

1. **Database Persistence**: Config stored correctly
2. **JSON Update Methods**: Both Python and SQLite json_set work
3. **Config Read-back**: Saved config can be retrieved
4. **Multiple Update Scenarios**: Different configuration combinations

## Manual Testing

### Test Scenario 1: Futures Only (Recommended)

```bash
# Update account config
python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()
cursor.execute("SELECT config FROM account_config WHERE user_id = 'gz_user1'")
row = cursor.fetchone()
if row:
    config = json.loads(row[0])
    config['enable_spot'] = False
    cursor.execute("UPDATE account_config SET config = ? WHERE user_id = 'gz_user1'",
                   (json.dumps(config),))
    conn.commit()
conn.close()
EOF

# Restart TD
pm2 restart td_binance:gz_user1

# Check logs
pm2 logs td_binance:gz_user1 --lines 20
```

**Expected Log Output**:
```
[info] Connecting BINANCE TD for gz_user1 (Spot: disabled, Futures: enabled)
[info] Futures market disabled by configuration
[info] Skipping Spot initialization (disabled or client unavailable)
[info] login success
```

**Should NOT See**:
```
[error] spot login failed, error_id: -2015
```

### Test Scenario 2: Verify Backward Compatibility

```bash
# Remove enable_spot flag
python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()
cursor.execute("SELECT config FROM account_config WHERE user_id = 'gz_user1'")
row = cursor.fetchone()
if row:
    config = json.loads(row[0])
    config.pop('enable_spot', None)  # Remove flag
    config.pop('enable_futures', None)
    cursor.execute("UPDATE account_config SET config = ? WHERE user_id = 'gz_user1'",
                   (json.dumps(config),))
    conn.commit()
conn.close()
EOF

# Restart TD
pm2 restart td_binance:gz_user1

# Check logs
pm2 logs td_binance:gz_user1 --lines 20
```

**Expected**: Both markets should initialize (default behavior)

## Troubleshooting

### Test Compilation Fails

```bash
# Ensure GoogleTest is available
ls core/deps/googletest-1.9.0/

# Clean build
cd core/build
rm -rf *
cmake ..
make
```

### Integration Test Fails

```bash
# Check database exists
ls -la /root/.config/kungfu/app/kungfu.db

# Check PM2 is installed
pm2 --version

# Run test with verbose output
bash -x ./integration_test_market_toggle.sh
```

## Design Patterns Used

1. **Guard Clause Pattern**: Early returns with boolean checks
2. **Strategy Pattern**: Configuration determines runtime behavior
3. **Fail-Safe Defaults**: True defaults prevent breaking changes
4. **Test-Driven Development**: Tests written before implementation

## Related Documentation

- [ADR-004](../../../doc/adr/004-binance-market-toggle.md) - Design decision record
- [TESTNET.md](../../../doc/TESTNET.md) - User-facing documentation
- [Example Configs](../../../doc/examples/binance_market_toggle_config.json)


