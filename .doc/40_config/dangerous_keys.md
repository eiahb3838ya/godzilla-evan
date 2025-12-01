---
title: Dangerous Configuration Keys
updated_at: 2025-11-17
owner: core-dev
lang: en
tokens_estimate: 3000
layer: 40_config
tags: [security, config, api-keys, risk, testnet, production]
purpose: "Documents high-risk configuration keys and security considerations"
---

# Dangerous Configuration Keys

**WARNING**: This document identifies high-risk configuration keys that can cause financial loss, security breaches, or production incidents if mishandled.

## Table of Contents

- [Severity Levels](#severity-levels)
- [Critical Keys](#critical-keys)
- [High-Risk Keys](#high-risk-keys)
- [Medium-Risk Keys](#medium-risk-keys)
- [Best Practices](#best-practices)

## Severity Levels

| Level | Description | Example Impact |
|-------|-------------|----------------|
| **CRITICAL** | Can cause direct financial loss or security breach | Production API keys exposed, wrong environment keys |
| **HIGH** | Can cause system failure or data corruption | Incorrect market toggle, missing required keys |
| **MEDIUM** | Can cause operational issues or degraded functionality | Logging level misconfiguration |

## Critical Keys

### `access_key` {#access_key}

**Severity**: 🔴 CRITICAL

**Why Dangerous**:
1. **Financial Loss**: Compromised production keys enable unauthorized trading
2. **API Abuse**: Leaked keys can be used to exhaust rate limits
3. **Security Breach**: Keys provide full account access

**Code Locations**:
- Definition: [core/extensions/binance/include/common.h:21](../../core/extensions/binance/include/common.h#L21)
- Usage: See [config_usage_map.md#access_key](config_usage_map.md#access_key)

**Common Mistakes**:

❌ **NEVER DO THIS**:
```json
{
  "access_key": "production_key_123",  // ← Production key in testnet config!
  "secret_key": "production_secret"
}
```

❌ **NEVER COMMIT KEYS TO GIT**:
```bash
# Bad: Keys in config file in repository
git add account_config.json  # ← Contains real API keys!
```

✅ **DO THIS**:
```json
{
  "access_key": "testnet_key_abc",  // ← Clearly marked as testnet
  "secret_key": "testnet_secret"
}
```

**Safety Checklist**:
- [ ] Never commit real API keys to version control
- [ ] Use testnet keys for development (get from https://testnet.binance.vision/key/generate)
- [ ] Use environment-specific key naming (prefix with `TESTNET_` or `PROD_`)
- [ ] Rotate production keys regularly
- [ ] Monitor API key usage for anomalies
- [ ] Restrict API key permissions (trading-only, no withdrawals)

**Recovery Procedures**:
1. **If production key leaked**:
   - Immediately revoke the key on Binance
   - Generate new key with minimal permissions
   - Audit all recent trades
   - Review git history: `git log -p | grep -i "api_key"`

2. **If key committed to git**:
   ```bash
   # Remove from git history (use with caution)
   git filter-branch --tree-filter 'rm -f config.json' HEAD
   # Force push (coordinate with team first!)
   git push --force
   ```

### `secret_key` {#secret_key}

**Severity**: 🔴 CRITICAL

**Why Dangerous**:
1. **Authentication Bypass**: Secret key is used to sign all API requests
2. **Complete Account Control**: With access_key + secret_key, attacker has full control
3. **Irreversible Actions**: Can execute trades, modify orders, access account info

**Code Locations**:
- Definition: [core/extensions/binance/include/common.h:22](../../core/extensions/binance/include/common.h#L22)
- Usage: See [config_usage_map.md#secret_key](config_usage_map.md#secret_key)

**Common Mistakes**:

❌ **NEVER LOG SECRET KEYS**:
```cpp
// BAD: Logging secret key
LOG_INFO("Config loaded: secret_key=" + config_.secret_key);  // ← NEVER DO THIS
```

❌ **NEVER EXPOSE IN ERROR MESSAGES**:
```cpp
// BAD: Including secret in error
throw std::runtime_error("Failed with secret: " + secret_key);  // ← NEVER DO THIS
```

✅ **DO THIS**:
```cpp
// GOOD: Log only key prefix for debugging
LOG_INFO("Config loaded: secret_key=" + config_.secret_key.substr(0, 4) + "****");
```

**Safety Checklist**:
- [ ] Never log full secret keys (log only first 4 chars + masking)
- [ ] Never include secrets in error messages
- [ ] Store secrets encrypted at rest
- [ ] Use secure key management (consider KMS for production)
- [ ] Separate testnet and production secrets completely
- [ ] Never reuse secrets across environments

**Current Code Review**:
✅ Code currently does NOT log secret keys (verified 2025-11-17)

## High-Risk Keys

### `enable_spot` {#enable_spot}

**Severity**: 🟠 HIGH

**Why Dangerous**:
1. **Testnet Incompatibility**: Wrong setting causes -2015 errors on Binance Testnet
2. **Failed Trades**: Disabling Spot when strategies expect it causes trade failures
3. **Hidden Failures**: Trader may silently skip Spot without obvious errors

**Code Locations**:
- Definition: [core/extensions/binance/include/common.h:26](../../core/extensions/binance/include/common.h#L26)
- Usage: See [config_usage_map.md#enable_spot](config_usage_map.md#enable_spot)

**Common Mistakes**:

❌ **MISTAKE**: Enabling Spot on Futures Testnet
```json
{
  "access_key": "futures_testnet_key",
  "secret_key": "futures_testnet_secret",
  "enable_spot": true,      // ← Will cause -2015 errors!
  "enable_futures": true
}
```

**Expected Error**:
```
[ERROR] Spot market login failed: -2015 Invalid API-key, IP, or permissions
```

✅ **CORRECT**: Disable Spot when using Futures Testnet keys
```json
{
  "access_key": "futures_testnet_key",
  "secret_key": "futures_testnet_secret",
  "enable_spot": false,     // ← Correct
  "enable_futures": true
}
```

**Safety Checklist**:
- [ ] Verify API key type matches enabled markets (Spot keys vs Futures keys)
- [ ] Test both markets before deploying strategies that use both
- [ ] Monitor logs for -2015 errors after config changes
- [ ] Document which strategies require which markets

**See Also**:
- [ADR-004: Binance Market Toggle](../95_adr/004-binance-market-toggle.md#context)
- [TESTNET.md: -2015 Error Resolution](../00_index/TESTNET.md)

### `enable_futures` {#enable_futures}

**Severity**: 🟠 HIGH

**Why Dangerous**:
1. **Leverage Risk**: Futures trading involves leverage and higher risk
2. **Failed Trades**: Disabling Futures when strategies expect it causes trade failures
3. **Liquidation Risk**: Wrong futures config can lead to unexpected liquidations

**Code Locations**:
- Definition: [core/extensions/binance/include/common.h:27](../../core/extensions/binance/include/common.h#L27)
- Usage: See [config_usage_map.md#enable_futures](config_usage_map.md#enable_futures)

**Common Mistakes**:

❌ **MISTAKE**: Forgetting Futures config in high-leverage strategy
```python
# Strategy expects Futures but config has it disabled
strategy = HighLeverageFuturesStrategy()  # ← Will fail silently
```

**Safety Checklist**:
- [ ] Verify strategy requirements match enabled markets
- [ ] Test with low leverage first (1x-2x)
- [ ] Set up liquidation alerts
- [ ] Never enable Futures without understanding leverage implications
- [ ] Monitor Futures positions actively

## Medium-Risk Keys

### Missing Required Keys

**Severity**: 🟡 MEDIUM

**Why Dangerous**:
- System fails to start without clear error message
- Silent failures can waste debugging time

**Required Keys**:
1. `access_key` - Always required
2. `secret_key` - Always required

**Error Behavior**:

If missing required keys, JSON deserialization will throw:
```cpp
// from_json() in common.h:46-47
j.at("access_key").get_to(c.access_key);  // ← Throws if missing
j.at("secret_key").get_to(c.secret_key);  // ← Throws if missing
```

**Validation**:
```bash
# Verify config has required keys
sqlite3 /app/runtime/db/account.db \
  "SELECT json_extract(config, '$.access_key'),
          json_extract(config, '$.secret_key')
   FROM account_config
   WHERE user_id = 'gz_user1';"
```

**Expected Output**:
```
testnet_key_abc|testnet_secret_xyz
```

If output is `NULL|NULL`, keys are missing!

## Best Practices

### 1. Environment Separation

**ALWAYS** keep testnet and production configurations completely separate:

```
Production:
- Database: /app/runtime/prod/db/account.db
- Keys: prod_api_key, prod_api_secret
- Enable: Both markets enabled for production trading

Testnet:
- Database: /app/runtime/testnet/db/account.db
- Keys: testnet_api_key, testnet_api_secret
- Enable: enable_spot=false for Futures Testnet
```

### 2. Key Naming Convention

Use clear prefixes:

✅ **GOOD**:
- `TESTNET_SPOT_KEY_2025`
- `TESTNET_FUTURES_KEY_2025`
- `PROD_SPOT_KEY_2025`
- `PROD_FUTURES_KEY_2025`

❌ **BAD**:
- `key123`
- `binance_key`
- `my_api_key`

### 3. Configuration Validation Script

Create a validation script before starting traders:

```bash
#!/bin/bash
# validate_config.sh

CONFIG=$(sqlite3 /app/runtime/db/account.db \
  "SELECT config FROM account_config WHERE user_id = 'gz_user1'")

# Check for required keys
if ! echo "$CONFIG" | grep -q "access_key"; then
  echo "ERROR: access_key missing!"
  exit 1
fi

if ! echo "$CONFIG" | grep -q "secret_key"; then
  echo "ERROR: secret_key missing!"
  exit 1
fi

# Check for testnet keys in production (example)
if [[ "$ENV" == "production" ]]; then
  if echo "$CONFIG" | grep -qi "testnet"; then
    echo "ERROR: Testnet keys detected in production config!"
    exit 1
  fi
fi

echo "Config validation passed"
```

### 4. Access Control

Limit database access:

```bash
# Only app user should read account.db
chmod 600 /app/runtime/db/account.db
chown app:app /app/runtime/db/account.db
```

### 5. Monitoring and Alerts

Set up alerts for:
- Failed login attempts (wrong keys)
- -2015 errors (wrong market enabled)
- Unexpected API key usage patterns
- Rate limit warnings

### 6. Incident Response Plan

1. **If wrong environment keys used**:
   - Immediately stop all traders
   - Verify no real trades executed
   - Replace keys with correct environment
   - Restart traders
   - Monitor for 1 hour

2. **If API keys compromised**:
   - Revoke keys immediately on Binance
   - Audit all trades in past 24 hours
   - Generate new keys with IP restrictions
   - Update config with new keys
   - Review access logs

## Related Documentation

- [config_usage_map.md](config_usage_map.md) - Complete config key reference
- [TESTNET.md](../00_index/TESTNET.md) - Testnet configuration guide
- [ADR-004: Binance Market Toggle](../95_adr/004-binance-market-toggle.md) - Market toggle feature
- [DEBUGGING.md: Config Issues](../85_memory/DEBUGGING.md) - Config troubleshooting

## Changelog

- **2025-11-17**: Initial creation documenting all dangerous config keys
- **2025-11-17**: Added enable_spot and enable_futures risk analysis

---

**Security Notice**: This document should be reviewed whenever new configuration keys are added. Always assess security and financial risk implications.
