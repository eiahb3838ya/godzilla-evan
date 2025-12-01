---
title: Binance Configuration Contract
updated_at: 2025-11-17
owner: core-dev
lang: en
tags: [contract, config, binance, credentials, endpoints, security]
code_refs:
  - core/extensions/binance/include/common.h:18-71
  - core/extensions/binance/package.json
purpose: "Defines Binance account configuration schema, defaults, and security considerations"
---

# Binance Configuration Contract

## Purpose

Defines the complete configuration schema for Binance exchange connectivity, including credentials, market toggles, and endpoint overrides. This contract specifies mandatory fields, defaults, and security implications of each configuration key.

## Configuration Structure

**Source:** [core/extensions/binance/include/common.h:18-71](../../core/extensions/binance/include/common.h)

### Schema Definition

```cpp
struct Configuration {
    // Authentication (REQUIRED)
    std::string user_id;
    std::string access_key;
    std::string secret_key;

    // Market Toggle Flags (ADR-004)
    bool enable_spot = true;      // Default: enabled
    bool enable_futures = true;   // Default: enabled

    // Spot Market Endpoints
    std::string spot_rest_host;
    int spot_rest_port;
    std::string spot_wss_host;
    int spot_wss_port;

    // USDT-Margined Futures (U-based)
    std::string ubase_rest_host;
    int ubase_rest_port;
    std::string ubase_wss_host;
    int ubase_wss_port;

    // Coin-Margined Futures (C-based)
    std::string cbase_rest_host;
    int cbase_rest_port;
    std::string cbase_wss_host;
    int cbase_wss_port;
};
```

## Field Specifications

### Authentication Fields (MANDATORY)

| Field | Type | Required | Security | Description |
|-------|------|----------|----------|-------------|
| `user_id` | string | Yes | Low | User identifier (informational only) |
| `access_key` | string | Yes | **CRITICAL** | Binance API key (public part) |
| `secret_key` | string | Yes | **CRITICAL** | Binance API secret (private key for HMAC signing) |

**Security Level: CRITICAL**
- **Never commit to version control**
- **Never log in plaintext**
- **Rotate regularly** (especially after suspected compromise)
- **Use testnet keys for development/testing**

See [Dangerous Configuration Keys](../40_config/CONFIG_REFERENCE.md#part-2-security-guidelines#binance-credentials) for detailed security practices.

### Market Toggle Flags (OPTIONAL - ADR-004)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `enable_spot` | bool | No | `true` | Enable spot market data and trading |
| `enable_futures` | bool | No | `true` | Enable USDT-margined futures |

**Purpose:** Runtime control over which markets to activate.

**Use Cases:**
- **Testnet limitations:** Futures testnet requires separate API keys
- **Credential isolation:** Use spot-only keys to prevent futures access
- **Performance:** Reduce subscription load by disabling unused markets
- **Compliance:** Disable futures in jurisdictions where derivatives are restricted

**Implementation:** See [ADR-004: Binance Market Toggle](../95_adr/004-binance-market-toggle.md)

**Backward Compatibility:** Both default to `true` to preserve existing behavior.

### Endpoint Configuration (OPTIONAL)

All endpoint fields are **optional**. If not specified in config, hardcoded testnet defaults are used.

#### Spot Market Endpoints

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `spot_rest_host` | string | `"testnet.binance.vision"` | Spot REST API host |
| `spot_rest_port` | int | `443` | Spot REST API port (HTTPS) |
| `spot_wss_host` | string | `"stream.testnet.binance.vision"` | Spot WebSocket host |
| `spot_wss_port` | int | `443` | Spot WebSocket port (WSS) |

**Production Overrides (NOT in code):**
```json
{
  "spot_rest_host": "api.binance.com",
  "spot_wss_host": "stream.binance.com"
}
```

#### USDT-Margined Futures Endpoints (U-based)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `ubase_rest_host` | string | `"testnet.binancefuture.com"` | Futures REST API host |
| `ubase_rest_port` | int | `443` | Futures REST API port (HTTPS) |
| `ubase_wss_host` | string | `"stream.binancefuture.com"` | Futures WebSocket host |
| `ubase_wss_port` | int | `443` | Futures WebSocket port (WSS) |

**Production Overrides (NOT in code):**
```json
{
  "ubase_rest_host": "fapi.binance.com",
  "ubase_wss_host": "fstream.binance.com"
}
```

#### Coin-Margined Futures Endpoints (C-based)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cbase_rest_host` | string | `"testnet.binancefuture.com"` | Coin-margined REST API host |
| `cbase_rest_port` | int | `443` | Coin-margined REST API port |
| `cbase_wss_host` | string | `"dstream.binancefuture.com"` | Coin-margined WebSocket host |
| `cbase_wss_port` | int | `443` | Coin-margined WebSocket port |

**Production Overrides (NOT in code):**
```json
{
  "cbase_rest_host": "dapi.binance.com",
  "cbase_wss_host": "dstream.binance.com"
}
```

**Note:** Coin-margined futures are NOT currently used by the system (USDT-margined only).

## Configuration Loading

### JSON Deserialization

**Source:** [core/extensions/binance/include/common.h:43-71](../../core/extensions/binance/include/common.h)

```cpp
inline void from_json(const nlohmann::json &j, Configuration &c)
{
    // REQUIRED fields - will throw if missing
    j.at("user_id").get_to(c.user_id);
    j.at("access_key").get_to(c.access_key);
    j.at("secret_key").get_to(c.secret_key);

    // OPTIONAL fields - use defaults if missing (ADR-004)
    c.enable_spot = j.value("enable_spot", true);
    c.enable_futures = j.value("enable_futures", true);

    // Endpoint defaults (hardcoded to testnet)
    c.spot_rest_host = "testnet.binance.vision";
    c.spot_rest_port = 443;
    // ... (all other endpoints)
}
```

**Critical:** Endpoint fields are NOT read from JSON - they are hardcoded to testnet defaults. To use production, code must be modified.

## Configuration Examples

### Minimal Configuration (Testnet Spot Only)

```json
{
  "user_id": "test_user_001",
  "access_key": "your_spot_testnet_api_key_here",
  "secret_key": "your_spot_testnet_secret_here",
  "enable_spot": true,
  "enable_futures": false
}
```

**Result:**
- Spot market enabled on testnet
- Futures disabled (no subscription, no trading)
- All endpoints use hardcoded testnet defaults

### Testnet with Both Markets

```json
{
  "user_id": "test_user_001",
  "access_key": "your_futures_testnet_api_key_here",
  "secret_key": "your_futures_testnet_secret_here",
  "enable_spot": true,
  "enable_futures": true
}
```

**Requirements:**
- Must use Futures Testnet API key (different from Spot Testnet)
- See [TESTNET.md](../00_index/TESTNET.md) for key generation

### Production Configuration (REQUIRES CODE CHANGE)

```json
{
  "user_id": "prod_user_001",
  "access_key": "PROD_API_KEY_NEVER_COMMIT",
  "secret_key": "PROD_SECRET_NEVER_COMMIT",
  "enable_spot": true,
  "enable_futures": true
}
```

**Plus manual code edit in common.h:**

```cpp
// Change lines 56-70 from testnet to production
c.spot_rest_host = "api.binance.com";           // was testnet.binance.vision
c.spot_wss_host = "stream.binance.com";         // was stream.testnet.binance.vision
c.ubase_rest_host = "fapi.binance.com";         // was testnet.binancefuture.com
c.ubase_wss_host = "fstream.binance.com";       // was stream.binancefuture.com
```

**Warning:** This requires recompilation and is error-prone. Future enhancement should make endpoints configurable via JSON.

## Validation Rules

### 1. Credential Format
- `access_key`: 64-character alphanumeric string
- `secret_key`: 64-character alphanumeric string
- `user_id`: any non-empty string (not validated by system)

### 2. Market Toggle Combinations

| enable_spot | enable_futures | Valid? | Notes |
|-------------|----------------|--------|-------|
| `true` | `true` | ✅ Yes | Both markets active (requires compatible API key) |
| `true` | `false` | ✅ Yes | Spot only |
| `false` | `true` | ✅ Yes | Futures only |
| `false` | `false` | ⚠️ Warning | No markets enabled (system starts but no data) |

### 3. Endpoint Consistency
- All `*_rest_port` and `*_wss_port` should be `443` (HTTPS/WSS standard)
- All `*_rest_host` and `*_wss_host` must be valid DNS hostnames
- Ports are `int` type (not string)

## Security Considerations

### Credential Leakage Vectors

**HIGH RISK:**
1. **Git commits** - Never commit configs with real keys
2. **Logs** - Ensure logging does not print `access_key` or `secret_key`
3. **Error messages** - Truncate keys in exception messages
4. **Core dumps** - Secrets remain in memory

**Mitigation:**
- Use `.gitignore` for all `*.json` config files
- Store production configs outside git repository
- Use environment variables for CI/CD: `BINANCE_ACCESS_KEY`, `BINANCE_SECRET_KEY`
- Implement key rotation policy (every 90 days)

### Testnet vs Production Keys

| Environment | Key Type | Risk | Storage |
|-------------|----------|------|---------|
| Testnet | Spot Testnet API key | Low | Can commit to **private** repos |
| Testnet | Futures Testnet API key | Low | Can commit to **private** repos |
| Production | Production API key | **CRITICAL** | **NEVER commit**, use secrets manager |

**Testnet Safety:**
- No real funds at risk
- Keys can be regenerated freely
- Rate limits are lower than production

**Production Safety:**
- Real funds accessible
- Withdrawal permissions controllable (disable on API key)
- IP whitelist recommended
- Use read-only keys for market data-only bots

## Related Documentation

### Configuration
- [Config Usage Map](../40_config/CONFIG_REFERENCE.md) - Maps all config keys to code
- [Dangerous Configuration Keys](../40_config/CONFIG_REFERENCE.md#part-2-security-guidelines) - Security deep dive

### Deployment
- [TESTNET.md](../00_index/TESTNET.md) - Complete testnet setup guide
- [INSTALL.md](../00_index/INSTALL.md) - Environment setup

### Implementation
- [Binance Extension Module](../10_modules/binance_extension.md) - How config is consumed
- [ADR-004: Binance Market Toggle](../95_adr/004-binance-market-toggle.md) - Design rationale for `enable_spot`/`enable_futures`

## Configuration File Locations

### Runtime Config
```
KF_HOME/
└── runtime/
    └── td/
        └── binance/
            └── <account_id>/
                └── config.json    # Account-specific config
```

### Interactive Setup
```bash
# CLI wizard creates config interactively
kfc account add binance my_account

# Prompts for:
# - User ID
# - Access key (hidden input)
# - Secret key (hidden input)
# - Enable spot? (y/n)
# - Enable futures? (y/n)
```

See [CLI Operations Guide](../90_operations/cli_operations_guide.md) for command reference.

## Future Enhancements

### Proposed: JSON-Configurable Endpoints
Currently endpoints are hardcoded. Future versions should support:

```json
{
  "user_id": "prod_user",
  "access_key": "...",
  "secret_key": "...",
  "enable_spot": true,
  "enable_futures": true,
  "endpoints": {
    "spot_rest_host": "api.binance.com",
    "spot_wss_host": "stream.binance.com",
    "futures_rest_host": "fapi.binance.com",
    "futures_wss_host": "fstream.binance.com"
  }
}
```

**Benefits:**
- No code changes for production deployment
- Easy regional endpoint switching (US, EU, Asia)
- Testnet/production toggle via config only

**Blocker:** Requires refactoring `from_json()` to parse optional `endpoints` object.

## Version History

- **2025-11-17:** Initial contract documentation
- **2025-03-03:** Added `enable_spot`/`enable_futures` flags (ADR-004)
- **2025-03-03:** Hardcoded testnet endpoints (original implementation)
