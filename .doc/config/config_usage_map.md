---
title: Configuration Usage Map
updated_at: 2025-11-24
owner: core-dev
lang: en
tokens_estimate: 5200
layer: 40_config
tags: [config, reference, binance, api-keys, mapping, cli, database, account-naming]
code_refs:
  - core/python/kungfu/command/account/add.py:15-25
  - core/python/kungfu/command/td.py:22-23
  - core/python/kungfu/data/sqlite/models.py:23-28
  - core/python/kungfu/data/sqlite/data_proxy.py:80-82
  - core/extensions/binance/package.json:8-44
purpose: "Maps all configuration keys to code locations and documents their usage"
---

# Configuration Usage Map

Complete mapping of all configuration keys to their code locations and usage patterns. Use this document to understand how config changes affect the system.

## Table of Contents

- [Binance Extension Configuration](#binance-extension-configuration)
- [System Configuration](#system-configuration)
- [Configuration Storage](#configuration-storage)
- [How to Update Configuration](#how-to-update-configuration)

## Binance Extension Configuration

### Overview

Binance extension configuration is stored in the account database and consumed by C++ code at runtime.

**Package Definition**: [core/extensions/binance/package.json](../../core/extensions/binance/package.json)

**Config Struct**: [core/extensions/binance/include/common.h:19-53](../../core/extensions/binance/include/common.h#L19-L53)

### Configuration Keys

#### `access_key` (required)

- **Type**: string
- **Purpose**: Binance API access key for authentication
- **Definition**: [core/extensions/binance/include/common.h:21](../../core/extensions/binance/include/common.h#L21)
- **Package**: [core/extensions/binance/package.json:16-20](../../core/extensions/binance/package.json#L16-L20)
- **Usage Locations**:
  - Market data initialization: [core/extensions/binance/src/marketdata_binance.cpp:34,42](../../core/extensions/binance/src/marketdata_binance.cpp#L34)
  - Spot trader initialization: [core/extensions/binance/src/trader_binance.cpp:56](../../core/extensions/binance/src/trader_binance.cpp#L56)
  - Futures trader initialization: [core/extensions/binance/src/trader_binance.cpp:72](../../core/extensions/binance/src/trader_binance.cpp#L72)
  - WebSocket reconnection: [core/extensions/binance/src/trader_binance.cpp:140,184](../../core/extensions/binance/src/trader_binance.cpp#L140)

**Security Note**: See [dangerous_keys.md](dangerous_keys.md#access_key) for security considerations.

#### `secret_key` (required)

- **Type**: string
- **Purpose**: Binance API secret key for request signing
- **Definition**: [core/extensions/binance/include/common.h:22](../../core/extensions/binance/include/common.h#L22)
- **Package**: [core/extensions/binance/package.json:24-28](../../core/extensions/binance/package.json#L24-L28)
- **Usage Locations**:
  - Market data initialization: [core/extensions/binance/src/marketdata_binance.cpp:35,43](../../core/extensions/binance/src/marketdata_binance.cpp#L35)
  - Spot trader initialization: [core/extensions/binance/src/trader_binance.cpp:57](../../core/extensions/binance/src/trader_binance.cpp#L57)
  - Futures trader initialization: [core/extensions/binance/src/trader_binance.cpp:73](../../core/extensions/binance/src/trader_binance.cpp#L73)
  - WebSocket reconnection: [core/extensions/binance/src/trader_binance.cpp:141,185](../../core/extensions/binance/src/trader_binance.cpp#L141)

**Security Note**: See [dangerous_keys.md](dangerous_keys.md#secret_key) for security considerations.

#### `enable_spot` (optional)

- **Type**: boolean
- **Default**: `true`
- **Purpose**: Enable/disable Spot market connections
- **Introduced**: ADR-004 (Market Toggle Feature)
- **Definition**: [core/extensions/binance/include/common.h:26](../../core/extensions/binance/include/common.h#L26)
- **Package**: [core/extensions/binance/package.json:31-36](../../core/extensions/binance/package.json#L31-L36)
- **Parsing**: [core/extensions/binance/include/common.h:51](../../core/extensions/binance/include/common.h#L51)
- **Usage Locations**:
  - Trader initialization check: [core/extensions/binance/src/trader_binance.cpp:51](../../core/extensions/binance/src/trader_binance.cpp#L51)
  - Login attempt check: [core/extensions/binance/src/trader_binance.cpp:112](../../core/extensions/binance/src/trader_binance.cpp#L112)
  - WebSocket reconnection check: [core/extensions/binance/src/trader_binance.cpp:372](../../core/extensions/binance/src/trader_binance.cpp#L372)
  - Logging: [core/extensions/binance/src/trader_binance.cpp:101](../../core/extensions/binance/src/trader_binance.cpp#L101)

**Use Case**: Set to `false` when using Futures Testnet to avoid -2015 errors from Spot endpoints.

**Example**:
```json
{
  "access_key": "YOUR_FUTURES_API_KEY",
  "secret_key": "YOUR_FUTURES_SECRET_KEY",
  "enable_spot": false,
  "enable_futures": true
}
```

**See Also**:
- [ADR-004: Binance Market Toggle](../95_adr/004-binance-market-toggle.md)
- [Config Example](../96_examples/binance_market_toggle_config.json)
- [TESTNET.md: Disabling Spot Login](../00_index/TESTNET.md)

#### `enable_futures` (optional)

- **Type**: boolean
- **Default**: `true`
- **Purpose**: Enable/disable Futures market connections
- **Introduced**: ADR-004 (Market Toggle Feature)
- **Definition**: [core/extensions/binance/include/common.h:27](../../core/extensions/binance/include/common.h#L27)
- **Package**: [core/extensions/binance/package.json:38-43](../../core/extensions/binance/package.json#L38-L43)
- **Parsing**: [core/extensions/binance/include/common.h:52](../../core/extensions/binance/include/common.h#L52)
- **Usage Locations**:
  - Trader initialization check: [core/extensions/binance/src/trader_binance.cpp:67](../../core/extensions/binance/src/trader_binance.cpp#L67)
  - Login attempt check: [core/extensions/binance/src/trader_binance.cpp:106](../../core/extensions/binance/src/trader_binance.cpp#L106)
  - WebSocket reconnection check: [core/extensions/binance/src/trader_binance.cpp:376](../../core/extensions/binance/src/trader_binance.cpp#L376)
  - Logging: [core/extensions/binance/src/trader_binance.cpp:102](../../core/extensions/binance/src/trader_binance.cpp#L102)

**Use Case**: Set to `false` when only trading Spot to reduce unnecessary connections.

**Example**:
```json
{
  "access_key": "YOUR_SPOT_API_KEY",
  "secret_key": "YOUR_SPOT_SECRET_KEY",
  "enable_spot": true,
  "enable_futures": false
}
```

**See Also**:
- [ADR-004: Binance Market Toggle](../95_adr/004-binance-market-toggle.md)
- [Config Example](../96_examples/binance_market_toggle_config.json)

### Endpoint Configuration (Hardcoded Defaults)

**Important**: The following endpoint configurations are **NOT configurable via JSON**. They are hardcoded in the `from_json` function and always use testnet values. To use production endpoints, source code must be modified.

**Source**: [core/extensions/binance/include/common.h:54-71](../../core/extensions/binance/include/common.h#L54-L71)

#### Spot Market Endpoints

| Field | Type | Default (Testnet) | Production Value | Purpose |
|-------|------|-------------------|------------------|---------|
| `spot_rest_host` | string | `"testnet.binance.vision"` | `"api.binance.com"` | Spot REST API hostname |
| `spot_rest_port` | int | `443` | `443` | Spot REST API port (HTTPS) |
| `spot_wss_host` | string | `"stream.testnet.binance.vision"` | `"stream.binance.com"` | Spot WebSocket hostname |
| `spot_wss_port` | int | `443` | `443` | Spot WebSocket port (WSS) |

**Definition**: [core/extensions/binance/include/common.h:29-32](../../core/extensions/binance/include/common.h#L29-L32)
**Hardcoded Values**: [core/extensions/binance/include/common.h:56-59](../../core/extensions/binance/include/common.h#L56-L59)

#### USDT-Margined Futures Endpoints (U-based)

| Field | Type | Default (Testnet) | Production Value | Purpose |
|-------|------|-------------------|------------------|---------|
| `ubase_rest_host` | string | `"testnet.binancefuture.com"` | `"fapi.binance.com"` | Futures REST API hostname |
| `ubase_rest_port` | int | `443` | `443` | Futures REST API port (HTTPS) |
| `ubase_wss_host` | string | `"stream.binancefuture.com"` | `"fstream.binance.com"` | Futures WebSocket hostname |
| `ubase_wss_port` | int | `443` | `443` | Futures WebSocket port (WSS) |

**Definition**: [core/extensions/binance/include/common.h:33-36](../../core/extensions/binance/include/common.h#L33-L36)
**Hardcoded Values**: [core/extensions/binance/include/common.h:63-66](../../core/extensions/binance/include/common.h#L63-L66)

#### Coin-Margined Futures Endpoints (C-based)

| Field | Type | Default (Testnet) | Production Value | Purpose |
|-------|------|-------------------|------------------|---------|
| `cbase_rest_host` | string | `"testnet.binancefuture.com"` | `"dapi.binance.com"` | Coin-margined REST API hostname |
| `cbase_rest_port` | int | `443` | `443` | Coin-margined REST API port |
| `cbase_wss_host` | string | `"dstream.binancefuture.com"` | `"dstream.binance.com"` | Coin-margined WebSocket hostname |
| `cbase_wss_port` | int | `443` | `443` | Coin-margined WebSocket port |

**Definition**: [core/extensions/binance/include/common.h:37-40](../../core/extensions/binance/include/common.h#L37-L40)
**Hardcoded Values**: [core/extensions/binance/include/common.h:67-70](../../core/extensions/binance/include/common.h#L67-L70)

**Note**: Coin-margined futures are currently NOT used by the system.

#### Switching to Production Endpoints

**Current Limitation**: Endpoints cannot be configured via JSON. To use production:

1. Edit [core/extensions/binance/include/common.h:54-71](../../core/extensions/binance/include/common.h#L54-L71)
2. Change hardcoded values:
   ```cpp
   // FROM (testnet):
   c.spot_rest_host = "testnet.binance.vision";
   c.spot_wss_host = "stream.testnet.binance.vision";
   c.ubase_rest_host = "testnet.binancefuture.com";
   c.ubase_wss_host = "stream.binancefuture.com";

   // TO (production):
   c.spot_rest_host = "api.binance.com";
   c.spot_wss_host = "stream.binance.com";
   c.ubase_rest_host = "fapi.binance.com";
   c.ubase_wss_host = "fstream.binance.com";
   ```
3. Recompile: `cd /app/core/build && make -j$(nproc)`

**Future Enhancement**: Make endpoints configurable via JSON to avoid code changes for production deployment. See [Binance Config Contract: Future Enhancements](../30_contracts/binance_config_contract.md#future-enhancements).

**See Also**:
- [Binance Configuration Contract](../30_contracts/binance_config_contract.md) - Complete endpoint specification
- [TESTNET.md](../00_index/TESTNET.md) - Testnet setup and endpoint documentation

---

## Strategy Configuration

### Overview

Strategy configuration is stored in JSON files within each strategy directory and loaded by the strategy runtime.

**Location**: `strategies/<strategy_name>/config.json`

**Loading**: [core/python/kungfu/wingchun/strategy.py:153-158](../../core/python/kungfu/wingchun/strategy.py#L153-L158)

### Configuration Keys

#### `symbol` (required)

- **Type**: string
- **Purpose**: Trading pair symbol for market data subscription and order placement
- **Format**: `lowercase_base_underscore_quote` (e.g., `"btc_usdt"`, `"eth_usdt"`)
- **Usage Locations**:
  - Market data subscription: [core/cpp/wingchun/src/strategy/context.cpp:264-271](../../core/cpp/wingchun/src/strategy/context.cpp#L264-L271)
  - Subscription matching: [core/cpp/wingchun/src/strategy/runner.cpp:72](../../core/cpp/wingchun/src/strategy/runner.cpp#L72)
  - Symbol hash generation: [core/cpp/wingchun/include/kungfu/wingchun/common.h:354-365](../../core/cpp/wingchun/include/kungfu/wingchun/common.h#L354-L365)
  - Base/quote extraction: [core/python/kungfu/wingchun/book/book.py:122-123](../../core/python/kungfu/wingchun/book/book.py#L122-L123)

**CRITICAL - Format Requirements**:

✓ **Correct Format**: `"btc_usdt"`, `"eth_usdt"`, `"sol_usdt"`
- Lowercase letters only
- Base and quote coins separated by single underscore

✗ **Wrong Formats**:
- `"btcusdt"` (no separator) → **IndexError** when placing orders
- `"BTCUSDT"` (uppercase) → **subscription mismatch**
- `"BTC_USDT"` (uppercase with underscore) → **subscription mismatch**
- `"btc-usdt"` (hyphen) → **IndexError**

**Why Format Matters**:

1. **Base/Quote Coin Extraction** ([book.py:122-123](../../core/python/kungfu/wingchun/book/book.py#L122-L123)):
   ```python
   splited = input.symbol.split("_")
   base_coin = splited[0]   # "btc_usdt" → "btc"
   quote_coin = splited[1]  # "btcusdt" → IndexError!
   ```

2. **Subscription Matching** ([common.h:354-365](../../core/cpp/wingchun/include/kungfu/wingchun/common.h#L354-L365)):
   ```cpp
   // Symbol string is hashed directly
   symbol_id = hash_str_32(symbol) ^ hash_str_32(sub_type) ^ ...
   // "btc_usdt" hash ≠ "btcusdt" hash → subscription fails silently
   ```

3. **Exchange Format Conversion** ([type_convert_binance.h:111-121](../../core/extensions/binance/include/type_convert_binance.h#L111-L121)):
   ```cpp
   // MD gateway converts internal → exchange format
   to_binance_symbol("btc_usdt") → "BTCUSDT" (for WebSocket)
   // Then publishes back with original format "btc_usdt"
   ```

**Example Configuration**:
```json
{
  "name": "demo_future",
  "md_source": "binance",
  "td_source": "binance",
  "symbol": "btc_usdt",
  "account": "gz_user1"
}
```

**Troubleshooting**:

If you see either of these errors, check symbol format:

1. **IndexError: list index out of range** at `book.py:123`
   - Fix: Change symbol to `"btc_usdt"` format
   - Requires C++ rebuild after fixing

2. **Strategy not receiving market data** (silent failure)
   - Symptom: `subscribe` succeeds but `on_depth()` never called
   - Fix: Check symbol format and rebuild C++
   - See [Debugging Guide](../90_operations/debugging_guide.md#問題-1-策略無法接收市場數據)

**See Also**:
- [Symbol Naming Convention](symbol_naming_convention.md) - Complete symbol format specification
- [Strategy Framework](../10_modules/strategy_framework.md) - Strategy configuration usage
- [Debugging Guide](../90_operations/debugging_guide.md) - Troubleshooting symbol format issues

---

## Configuration Storage

### Database Schema

Configurations are stored in the account database:

**Table**: `account_config`

**Schema**:
- `user_id` (text): User identifier
- `config` (JSON blob): Configuration JSON

**Location**: `KF_HOME/runtime/db/account.db`

**Note**: `KF_HOME` defaults to `/app/runtime` in Docker containers. See [DEBUGGING.md CLI-Only Rule](../85_memory/DEBUGGING.md) for KF_HOME handling.

### JSON Structure

Configuration is stored as JSON and deserialized using the `from_json` function:

**Deserialization**: [core/extensions/binance/include/common.h:44-53](../../core/extensions/binance/include/common.h#L44-L53)

```cpp
void from_json(const nlohmann::json& j, BinanceConfig& c) {
    j.at("access_key").get_to(c.access_key);
    j.at("secret_key").get_to(c.secret_key);

    // Optional fields with defaults
    c.enable_spot = j.value("enable_spot", true);
    c.enable_futures = j.value("enable_futures", true);
}
```

## How to Update Configuration

### Method 1: CLI Command (Recommended for Initial Setup)

**Source**: [core/python/kungfu/command/account/add.py:15-25](../../core/python/kungfu/command/account/add.py)

**Command**:
```bash
kfc account -s binance add
```

**Interactive Prompts** (based on [package.json:8-44](../../core/extensions/binance/package.json)):
```
请填写账户 user_id: gz_user1
请填写access_key: [hidden input]
请填写 secret_key: [hidden input]
是否启用现货市场登录？(true/false) [default: true]: false
是否启用期货市场登录？(true/false) [default: true]: true
```

**Result**:
- Account ID generated: `binance_gz_user1`
- Configuration stored in `accounts.db` table `account_config`

**Code Reference**:
```python
# core/python/kungfu/command/account/add.py:18-25
account_id = ctx.source + '_' + answers[ctx.schema['key']]  # "binance_gz_user1"
if find_account(ctx, account_id):
    click.echo('Duplicate account')
else:
    ctx.db.add_account(
        account_id=account_id,
        source_name=ctx.source,
        receive_md=receive_md,
        config=answers
    )
```

**⚠️ 重要：帳號命名機制**

系統使用兩套帳號名稱格式：
- **資料庫格式** (`account_id`)：`{source}_{account}` (如 `binance_gz_user1`)
- **運行時格式** (`account`)：純帳號名稱 (如 `gz_user1`)

當你輸入 `gz_user1` 時，系統會自動加上 `binance_` 前綴存入資料庫。但在以下場景中，你**必須使用純帳號名稱**（`gz_user1`）：
- TD gateway 啟動參數：`-a gz_user1`
- 策略配置檔 `config.json`：`"account": "gz_user1"`
- 策略代碼：`context.add_account("binance", "gz_user1")`

詳細說明請參閱 [帳號命名機制](account_naming_convention.md)。

### Method 2: Python Script Update (For Modifying Existing Accounts)

**Database Path**:
```
$KF_HOME/runtime/system/etc/kungfu/db/live/accounts.db
```

**View Configuration**:
```python
import sqlite3
import json

# Connect to database
db_path = '/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Query account config
cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_gz_user1'")
config_json = cursor.fetchone()[0]
config = json.loads(config_json)

print(json.dumps(config, indent=2))
conn.close()
```

**Update Configuration**:
```python
import sqlite3
import json

db_path = '/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Read current config
cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_gz_user1'")
config_json = cursor.fetchone()[0]
config = json.loads(config_json)

# Update config
config['enable_spot'] = False
config['enable_futures'] = True

# Write back
cursor.execute(
    "UPDATE account_config SET config = ? WHERE account_id = 'binance_gz_user1'",
    (json.dumps(config),)
)
conn.commit()
conn.close()
```

**Important**: Changes require restarting the trader process to take effect:
```bash
pm2 restart td_binance:gz_user1
```

### Method 3: SQLite Direct Update (Advanced)

**For systems with sqlite3 CLI available**:

```bash
# Update config
sqlite3 /home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db \
  "UPDATE account_config
   SET config = json_set(config, '$.enable_spot', json('false'))
   WHERE account_id = 'binance_gz_user1';"
```

**Verify**:
```bash
sqlite3 /home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db \
  "SELECT config FROM account_config WHERE account_id = 'binance_gz_user1';"
```

## Configuration Flow

```
┌─────────────────────┐
│  account.db         │
│  account_config     │
│  (JSON blob)        │
└──────────┬──────────┘
           │
           │ Read at trader startup
           │
           v
┌─────────────────────┐
│  from_json()        │
│  common.h:44-53     │
│  Deserialize JSON   │
└──────────┬──────────┘
           │
           │ Populate BinanceConfig struct
           │
           v
┌─────────────────────┐
│  BinanceConfig      │
│  common.h:19-27     │
│  - access_key       │
│  - secret_key       │
│  - enable_spot      │
│  - enable_futures   │
└──────────┬──────────┘
           │
           │ Used throughout trader lifecycle
           │
           v
┌─────────────────────┐
│  trader_binance.cpp │
│  - Initialization   │
│  - Login checks     │
│  - Reconnection     │
└─────────────────────┘
```

## Testing Configuration Changes

Configuration changes are tested in:

**Unit Tests**: [core/extensions/binance/test/test_market_toggle.cpp](../../core/extensions/binance/test/test_market_toggle.cpp)

**Integration Tests**: [core/extensions/binance/test/integration_test_market_toggle.sh](../../core/extensions/binance/test/integration_test_market_toggle.sh)

**Manual Testing Guide**: [core/extensions/binance/test/MANUAL_TEST_GUIDE.md](../../core/extensions/binance/test/MANUAL_TEST_GUIDE.md)

## Related Documentation

- [dangerous_keys.md](dangerous_keys.md) - High-risk configuration keys
- [ADR-004: Binance Market Toggle](../95_adr/004-binance-market-toggle.md) - Market toggle feature design
- [TESTNET.md](../00_index/TESTNET.md) - Testnet configuration guide
- [DEBUGGING.md](../85_memory/DEBUGGING.md) - Configuration troubleshooting cases
- [binance_market_toggle_config.json](../96_examples/binance_market_toggle_config.json) - Example configurations

## Changelog

- **2025-11-17**: Initial creation mapping all Binance config keys
- **2025-11-17**: Added enable_spot and enable_futures flags (ADR-004)

---

**Maintenance Note**: When adding new configuration keys:
1. Update this document with all code references
2. Add security analysis to [dangerous_keys.md](dangerous_keys.md) if applicable
3. Add examples to [96_examples/](../96_examples/)
4. Update relevant ADRs if introducing breaking changes
