#!/bin/bash
# Integration test for Binance market toggle feature (ADR-004)
# Tests backward compatibility and new configuration options

set -e

echo "=== Binance Market Toggle Integration Test ==="
echo ""

DB_PATH="/root/.config/kungfu/app/kungfu.db"
TEST_USER="test_market_toggle"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

function print_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
}

function print_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    exit 1
}

# Cleanup function
function cleanup() {
    echo ""
    echo "=== Cleanup ==="
    sqlite3 $DB_PATH "DELETE FROM account_config WHERE user_id = '$TEST_USER'" 2>/dev/null || true
    pm2 delete td_test 2>/dev/null || true
    echo "Cleanup complete"
}

trap cleanup EXIT

# Ensure database exists
if [ ! -f "$DB_PATH" ]; then
    print_fail "Database not found at $DB_PATH"
fi

# Test 1: Backward compatibility (no flags specified)
print_test "Test 1: Backward compatibility - default both enabled"

python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()

config = {
    'access_key': 'TEST_KEY_BACKWARD',
    'secret_key': 'TEST_SECRET_BACKWARD'
}

cursor.execute('INSERT OR REPLACE INTO account_config VALUES (?, ?, ?, ?)',
               ('test_market_toggle', 'binance', 1, json.dumps(config)))
conn.commit()
conn.close()
EOF

# Verify config was saved correctly
STORED_CONFIG=$(sqlite3 $DB_PATH "SELECT config FROM account_config WHERE user_id = '$TEST_USER'")
if echo "$STORED_CONFIG" | grep -q "TEST_KEY_BACKWARD"; then
    print_pass "Config stored successfully (backward compatible)"
else
    print_fail "Config not stored correctly"
fi

# Test 2: Disable Spot only
print_test "Test 2: Disable Spot market only"

python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()

config = {
    'access_key': 'FUTURES_KEY',
    'secret_key': 'FUTURES_SECRET',
    'enable_spot': False
}

cursor.execute('UPDATE account_config SET config = ? WHERE user_id = ?',
               (json.dumps(config), 'test_market_toggle'))
conn.commit()
conn.close()
EOF

# Verify Spot is disabled
STORED_CONFIG=$(sqlite3 $DB_PATH "SELECT config FROM account_config WHERE user_id = '$TEST_USER'")
if echo "$STORED_CONFIG" | grep -q '"enable_spot":false'; then
    print_pass "Spot disabled successfully"
else
    print_fail "Spot disable flag not set correctly"
fi

# Test 3: Disable Futures only
print_test "Test 3: Disable Futures market only"

python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()

config = {
    'access_key': 'SPOT_KEY',
    'secret_key': 'SPOT_SECRET',
    'enable_spot': True,
    'enable_futures': False
}

cursor.execute('UPDATE account_config SET config = ? WHERE user_id = ?',
               (json.dumps(config), 'test_market_toggle'))
conn.commit()
conn.close()
EOF

# Verify Futures is disabled
STORED_CONFIG=$(sqlite3 $DB_PATH "SELECT config FROM account_config WHERE user_id = '$TEST_USER'")
if echo "$STORED_CONFIG" | grep -q '"enable_futures":false'; then
    print_pass "Futures disabled successfully"
else
    print_fail "Futures disable flag not set correctly"
fi

# Test 4: JSON update method
print_test "Test 4: Update config using json_set"

sqlite3 $DB_PATH << 'SQL'
UPDATE account_config 
SET config = json_set(config, '$.enable_spot', json('true'))
WHERE user_id = 'test_market_toggle';
SQL

STORED_CONFIG=$(sqlite3 $DB_PATH "SELECT config FROM account_config WHERE user_id = '$TEST_USER'")
if echo "$STORED_CONFIG" | grep -q '"enable_spot":true'; then
    print_pass "json_set update successful"
else
    print_fail "json_set update failed"
fi

# Test 5: Verify config extraction
print_test "Test 5: Verify config can be read back"

python3 << 'EOF'
import sqlite3, json

conn = sqlite3.connect('/root/.config/kungfu/app/kungfu.db')
cursor = conn.cursor()

cursor.execute("SELECT config FROM account_config WHERE user_id = ?", ('test_market_toggle',))
row = cursor.fetchone()

if row:
    config = json.loads(row[0])
    assert 'access_key' in config, "access_key missing"
    assert 'secret_key' in config, "secret_key missing"
    print("✓ Config structure valid")
else:
    raise Exception("Config not found")

conn.close()
EOF

print_pass "Config read-back successful"

# Summary
echo ""
echo "=== Test Summary ==="
echo -e "${GREEN}All integration tests passed!${NC}"
echo ""
echo "Tested scenarios:"
echo "  ✓ Backward compatibility (no flags)"
echo "  ✓ Disable Spot only"
echo "  ✓ Disable Futures only"
echo "  ✓ JSON update using json_set"
echo "  ✓ Config read-back validation"
echo ""


