---
title: 帳號命名機制 (Account Naming Convention)
updated_at: 2025-11-24
owner: config-team
lang: zh-TW
tags: [config, account, naming, convention]
purpose: "說明系統中帳號名稱的兩套格式：資料庫格式 vs 運行時格式"
code_refs:
  - core/python/kungfu/command/account/add.py:18
  - core/python/kungfu/command/td.py:22-23
  - core/cpp/wingchun/src/broker/trader.cpp:27
  - core/cpp/wingchun/src/strategy/context.cpp:100-103
---

# 帳號命名機制

## 概覽

Godzilla 交易系統中的帳號名稱有**兩套格式**：

1. **資料庫格式** (`account_id`)：`{source}_{account}` (如 `binance_gz_user1`)
2. **運行時格式** (`account`)：純帳號名稱 (如 `gz_user1`)

理解這兩套格式的使用時機，對於正確配置策略和啟動服務至關重要。

---

## 命名格式對照表

| 使用場景 | 格式 | 範例 | 位置 |
|---------|------|------|------|
| 資料庫主鍵 (`account_id`) | `{source}_{account}` | `binance_gz_user1` | `accounts.db` 的 `account_config` 表 |
| TD gateway 啟動參數 (`-a`) | `{account}` | `gz_user1` | `pm2 start td_binance.json` |
| 策略配置 (`config.json`) | `{account}` | `gz_user1` | `strategies/*/config.json` |
| Location 註冊 (內部) | `td/{source}/{account}/live` | `td/binance/gz_user1/live` | C++ location 系統 |

---

## 完整流程追蹤

### 1. 創建帳號 (`kfc account -s binance add`)

當你執行交互式帳號創建時：

```bash
$ kfc account -s binance add
? Enter user_id: gz_user1
? Enter access_key: ****
? Enter secret_key: ****
```

**內部處理** ([add.py:18](core/python/kungfu/command/account/add.py#L18))：
```python
account_id = ctx.source + '_' + answers[ctx.schema['key']]  # "binance_gz_user1"
ctx.db.add_account(account_id=account_id, source_name=ctx.source, config=answers)
```

**結果**：
- 資料庫 `account_id`：`binance_gz_user1`
- 配置 `config.user_id`：`gz_user1`

---

### 2. 啟動 TD Gateway (`pm2 start td_binance.json`)

PM2 配置：
```json
{
  "apps": [{
    "name": "td_binance:gz_user1",
    "args": "-l info td -s binance -a gz_user1"
  }]
}
```

**內部處理** ([td.py:22-23](core/python/kungfu/command/td.py#L22-L23))：
```python
# 組合完整 account_id 查詢資料庫配置
account_config = ctx.db.get_td_account_config(source, source + '_' + account)
# → 查詢 "binance_gz_user1"

# 傳入純帳號名稱到 C++ extension
ext = EXTENSION_REGISTRY_TD.get_extension(source)(low_latency, ctx.locator, account, account_config)
# → 傳入 "gz_user1"
```

**C++ Location 註冊** ([trader.cpp:27](core/cpp/wingchun/src/broker/trader.cpp#L27))：
```cpp
Trader::Trader(bool low_latency, locator_ptr locator, const std::string &source, const std::string &account_id)
    : apprentice(location::make(mode::LIVE, category::TD, source, account_id, std::move(locator)))
    // → 註冊為 "td/binance/gz_user1/live"
```

**結果**：
- TD gateway 註冊的 location：`td/binance/gz_user1/live`
- UID：`9843dd4d` (hash 自動生成)

---

### 3. 策略調用 `add_account()`

**策略配置** (`strategies/demo_future/config.json`)：
```json
{
  "name": "demo_future",
  "td_source": "binance",
  "account": "gz_user1"  // ✓ 使用純帳號名稱
}
```

**策略代碼** (`demo_future.py`)：
```python
def pre_start(context):
    config = context.get_config()
    context.add_account(config["td_source"], config["account"])
    # → context.add_account("binance", "gz_user1")
```

**C++ 驗證邏輯** ([context.cpp:100-103](core/cpp/wingchun/src/strategy/context.cpp#L100-L103))：
```cpp
void Context::add_account(const std::string &source, const std::string &account)
{
    // 創建 location：td/binance/gz_user1/live
    auto account_location = location::make(mode::LIVE, category::TD, source, account, home->locator);

    // 檢查此 location 是否已被 TD gateway 註冊
    if (home->mode == mode::LIVE and not app_.has_location(account_location->uid)) {
        throw wingchun_error(fmt::format("invalid account {}@{}", account, source));
        // ✗ 如果使用 "binance_gz_user1"，location 會變成 td/binance/binance_gz_user1/live (不匹配！)
    }

    // ✓ 匹配成功，帳號添加完成
    accounts_[account_id] = account_location;
}
```

**結果**：
- ✓ 使用 `"account": "gz_user1"` → location 匹配 → 成功
- ✗ 使用 `"account": "binance_gz_user1"` → location 不匹配 → `invalid account binance_gz_user1@binance`

---

## 常見錯誤

### 錯誤 1: 策略配置使用完整 account_id

**錯誤配置**：
```json
{
  "account": "binance_gz_user1"  // ✗ 錯誤
}
```

**錯誤訊息**：
```
RuntimeError: invalid account binance_gz_user1@binance
```

**原因**：
策略嘗試查找 `td/binance/binance_gz_user1/live`，但 TD gateway 註冊的是 `td/binance/gz_user1/live`。

**修正**：
```json
{
  "account": "gz_user1"  // ✓ 正確
}
```

---

### 錯誤 2: TD gateway 參數與資料庫不匹配

**錯誤啟動**：
```bash
python3 dev_run.py -l info td -s binance -a binance_gz_user1  # ✗ 錯誤
```

**錯誤訊息**：
```
Account config not found: binance_binance_gz_user1
```

**原因**：
系統會組合成 `binance_binance_gz_user1` 去查詢資料庫。

**修正**：
```bash
python3 dev_run.py -l info td -s binance -a gz_user1  # ✓ 正確
```

---

## 最佳實踐

### ✓ 正確的帳號命名流程

1. **創建帳號時**，輸入純帳號名稱：
   ```bash
   $ kfc account -s binance add
   ? Enter user_id: gz_user1  # ✓ 不要加 binance_ 前綴
   ```

2. **啟動 TD gateway 時**，使用純帳號名稱：
   ```json
   {
     "args": "-l info td -s binance -a gz_user1"
   }
   ```

3. **策略配置中**，使用純帳號名稱：
   ```json
   {
     "account": "gz_user1"
   }
   ```

4. **策略代碼中**，直接使用配置：
   ```python
   context.add_account(config["td_source"], config["account"])
   ```

---

## 技術細節：為何需要兩套格式？

### 資料庫格式的必要性

資料庫需要全局唯一的主鍵，因此使用 `{source}_{account}` 格式：
- 允許不同交易所使用相同的帳號名稱
- 範例：`binance_user1` 和 `okx_user1` 可以共存

### 運行時格式的必要性

Location 系統已經包含 `source` 資訊（`td/{source}/{account}/live`），因此：
- 避免冗餘：不需要 `td/binance/binance_user1/live`
- 保持一致：所有交易所的 location 結構相同

---

## 資料庫查詢範例

檢查帳號配置：
```bash
docker-compose exec -T app python3 -c "
import sqlite3
conn = sqlite3.connect('/app/runtime/system/etc/kungfu/db/live/accounts.db')
cursor = conn.cursor()
cursor.execute('SELECT account_id, source_name FROM account_config')
for row in cursor.fetchall():
    print(f'account_id: {row[0]} | source_name: {row[1]}')
"
```

**預期輸出**：
```
account_id: binance_gz_user1 | source_name: binance
```

---

## Related Documentation

- [配置使用地圖](config_usage_map.md) - 配置檔位置與格式
- [CLI 操作指南](../90_operations/cli_operations_guide.md) - `kfc account` 命令詳解
- [策略框架](../10_modules/strategy_framework.md) - `add_account()` API 使用

---

## 常見問題 (FAQ)

**Q: 為什麼 `kfc account add` 會自動加上前綴？**
A: 為了確保資料庫中的 `account_id` 全局唯一，允許不同交易所使用相同的帳號名稱。

**Q: 我可以手動修改資料庫中的 `account_id` 嗎？**
A: ❌ 不建議。應該透過 `kfc account` 命令管理帳號，直接修改資料庫可能導致 TD gateway 無法正確載入配置。

**Q: 如果我想使用多個 Binance 帳號怎麼辦？**
A: 創建多個帳號時使用不同的名稱（如 `user1`, `user2`），系統會自動生成 `binance_user1`, `binance_user2`。

**Q: Location UID (`9843dd4d`) 是如何生成的？**
A: 透過 `yijinjing::util::hash_str_32()` 對 location 路徑進行 hash，確保每個 location 有唯一的 32-bit ID。

---

**版本**: 2025-11-24
**維護者**: config-team
**Token 估算**: ~2,800 tokens
