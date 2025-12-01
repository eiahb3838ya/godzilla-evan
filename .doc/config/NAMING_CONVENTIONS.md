---
title: 命名規範 (Naming Conventions)
updated_at: 2025-12-01
owner: config-team
lang: zh-TW
tokens_estimate: 4500
layer: config
tags: [config, naming, account, symbol, convention, troubleshooting]
purpose: "統一文檔: 帳號與交易對命名規範"
code_refs:
  - core/python/kungfu/command/account/add.py:18
  - core/python/kungfu/command/td.py:22-23
  - core/python/kungfu/wingchun/book/book.py:122-123
  - core/cpp/wingchun/include/kungfu/wingchun/common.h:354-365
  - core/extensions/binance/include/type_convert_binance.h:111-121
---

# 命名規範

本文檔統一說明系統中兩種關鍵命名規範：

1. **帳號命名** - 資料庫格式 vs 運行時格式
2. **交易對命名** - 格式要求與常見錯誤

---

## 一、帳號命名規範

### 概覽

Godzilla 交易系統中的帳號名稱有**兩套格式**:

1. **資料庫格式** (`account_id`): `{source}_{account}` (如 `binance_gz_user1`)
2. **運行時格式** (`account`): 純帳號名稱 (如 `gz_user1`)

理解這兩套格式的使用時機,對於正確配置策略和啟動服務至關重要。

### 命名格式對照表

| 使用場景 | 格式 | 範例 | 位置 |
|---------|------|------|------|
| 資料庫主鍵 (`account_id`) | `{source}_{account}` | `binance_gz_user1` | `accounts.db` 的 `account_config` 表 |
| TD gateway 啟動參數 (`-a`) | `{account}` | `gz_user1` | `pm2 start td_binance.json` |
| 策略配置 (`config.json`) | `{account}` | `gz_user1` | `strategies/*/config.json` |
| Location 註冊 (內部) | `td/{source}/{account}/live` | `td/binance/gz_user1/live` | C++ location 系統 |

### 完整流程追蹤

#### 1. 創建帳號 (`kfc account -s binance add`)

當你執行交互式帳號創建時:

```bash
$ kfc account -s binance add
? Enter user_id: gz_user1
? Enter access_key: ****
? Enter secret_key: ****
```

**內部處理** ([add.py:18](../../core/python/kungfu/command/account/add.py#L18)):
```python
account_id = ctx.source + '_' + answers[ctx.schema['key']]  # "binance_gz_user1"
ctx.db.add_account(account_id=account_id, source_name=ctx.source, config=answers)
```

**結果**:
- 資料庫 `account_id`: `binance_gz_user1`
- 配置 `config.user_id`: `gz_user1`
- 配置檔案: `~/.config/kungfu/app/runtime/config/td/binance/gz_user1.json`

**關鍵點**:
- ✓ 資料庫中用 `binance_gz_user1` (有 source 前綴)
- ✓ 配置檔案名用 `gz_user1.json` (無 source 前綴)
- ✓ 配置內容 `user_id` 字段是 `gz_user1`

#### 2. 啟動 TD Gateway (`pm2 start td_binance.json`)

PM2 配置檔案 (`scripts/binance_test/td_binance.json`):
```json
{
  "apps": [{
    "name": "td_binance",
    "args": "-l info td -s binance -a gz_user1",
              ⬆️ 使用運行時格式 (無 source 前綴)
  }]
}
```

**內部處理** ([td.py:22-23](../../core/python/kungfu/command/td.py#L22-L23)):
```python
account_id = f"{ctx.source}_{args.account}"  # 重新組裝為 "binance_gz_user1"
config = db.get_trader_config(account_id)    # 查詢資料庫
```

**關鍵點**:
- ✓ CLI 參數 `-a gz_user1` 是運行時格式
- ✓ 程式內部自動組裝成資料庫格式 `binance_gz_user1` 查詢

#### 3. 策略使用 (`context.add_account()`)

策略程式碼 (`strategies/demo_spot/demo_spot.py`):
```python
def pre_start(self, context):
    context.add_account("binance", "gz_user1")
                                  ⬆️ 使用運行時格式
```

**內部處理** ([context.cpp:100-103](../../core/cpp/wingchun/src/strategy/context.cpp#L100-L103)):
```cpp
std::string make_location_from_account(const std::string& source, const std::string& account) {
    return fmt::format("td/{}/{}/live", source, account);  // "td/binance/gz_user1/live"
}
```

**關鍵點**:
- ✓ `add_account()` 第二參數使用運行時格式 `gz_user1`
- ✓ Location 系統自動組裝為 `td/binance/gz_user1/live`

### 常見錯誤

#### 錯誤 1: PM2 使用資料庫格式

❌ **錯誤做法**:
```json
{
  "args": "-l info td -s binance -a binance_gz_user1"
                                  ⬆️ 錯誤! 有 source 前綴
}
```

**後果**: TD gateway 啟動失敗
```
ERROR: Account not found: binance_binance_gz_user1
                         ⬆️ 雙重前綴!
```

**原因**: `td.py` 會自動加上 source 前綴,變成 `binance_binance_gz_user1`

#### 錯誤 2: 策略使用資料庫格式

❌ **錯誤做法**:
```python
context.add_account("binance", "binance_gz_user1")
                              ⬆️ 錯誤! 有 source 前綴
```

**後果**: Location 註冊錯誤
```
ERROR: Cannot find location: td/binance/binance_gz_user1/live
                                      ⬆️ 多了一層 binance
```

**正確做法**: 使用運行時格式 `gz_user1`

#### 錯誤 3: 配置檔案名稱錯誤

❌ **錯誤做法**:
```bash
# 配置檔案路徑
~/.config/kungfu/app/runtime/config/td/binance/binance_gz_user1.json
                                              ⬆️ 錯誤! 有 source 前綴
```

**後果**: TD gateway 無法載入配置
```
ERROR: Config file not found: .../td/binance/binance_gz_user1.json
```

**正確路徑**: `~/.config/kungfu/app/runtime/config/td/binance/gz_user1.json`

### 快速參考表

| 你在做什麼 | 使用哪種格式 | 範例 |
|-----------|-------------|------|
| ✅ 創建帳號 (CLI `-a`) | 運行時格式 | `kfc account -s binance add` → 輸入 `gz_user1` |
| ✅ 啟動 TD (PM2 `args`) | 運行時格式 | `"-l info td -s binance -a gz_user1"` |
| ✅ 策略 `add_account()` | 運行時格式 | `context.add_account("binance", "gz_user1")` |
| ✅ 配置檔案名稱 | 運行時格式 | `~/.config/.../td/binance/gz_user1.json` |
| ✅ 查詢資料庫 (直接 SQL) | 資料庫格式 | `SELECT * FROM account_config WHERE account_id='binance_gz_user1'` |

---

## 二、交易對命名規範

### 概覽

Godzilla 交易系統中的交易對名稱有**嚴格的格式要求**:

**✓ 正確格式**: `小寫基幣_小寫報價幣` (如 `btc_usdt`, `eth_usdt`)  
**✗ 錯誤格式**: `btcusdt`, `BTCUSDT`, `BTC_USDT`, `btc-usdt`

使用錯誤的格式會導致:
1. **IndexError** 在下單時 (base/quote coin 解析失敗)
2. **靜默訂閱失敗** (策略無法收到市場數據)
3. **需要重新編譯 C++** 才能修復

### 為什麼格式這麼重要?

#### 1. Base/Quote Coin 解析 ([book.py:122-123](../../core/python/kungfu/wingchun/book/book.py#L122-L123))

當策略下單時,系統需要從交易對中提取基幣和報價幣:

```python
def on_order_input(self, input):
    # ...
    splited = input.symbol.split("_")
    base_coin = splited[0]   # ✓ "btc_usdt" → "btc"
    quote_coin = splited[1]  # ✓ "btc_usdt" → "usdt"
    # ...
```

**❌ 錯誤範例**:
```python
# 如果 symbol = "btcusdt" (沒有底線)
splited = "btcusdt".split("_")  # → ["btcusdt"] (只有 1 個元素)
base_coin = splited[0]          # → "btcusdt" (✓ 可存取)
quote_coin = splited[1]         # → IndexError: list index out of range (✗)
```

#### 2. 訂閱匹配機制 ([common.h:354-365](../../core/cpp/wingchun/include/kungfu/wingchun/common.h#L354-L365))

策略訂閱市場數據時,系統使用 **symbol hash** 來匹配事件:

```cpp
inline uint32_t get_symbol_id(const std::string &symbol, const std::string &sub_type,
                               const std::string &exchange) {
    return MurmurHash2(symbol.c_str(), symbol.length());
}
```

**❌ 錯誤範例**:
```python
# 策略訂閱
context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)
                            ⬆️ 錯誤! 沒有底線

# MD Gateway 收到的交易所數據
# Binance 回傳: "symbol": "BTCUSDT"
# 轉換後存入 Journal: "btcusdt" (全小寫)

# 匹配失敗!
# 策略訂閱的 hash("btcusdt") ≠ Journal 的 hash("btcusdt")
# (因為策略寫成 "btcusdt" 而非 "btc_usdt")
```

#### 3. 大小寫轉換 ([type_convert_binance.h:111-121](../../core/extensions/binance/include/type_convert_binance.h#L111-L121))

Binance Extension 會自動轉換大小寫:

```cpp
std::string convert_symbol_binance_to_kf(const std::string &binance_symbol) {
    std::string result = binance_symbol;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;  // "BTCUSDT" → "btcusdt"
}
```

**關鍵點**:
- ✓ 系統自動轉小寫,所以 `BTCUSDT` 和 `btcusdt` 等價
- ✗ 但系統**不會**自動加底線,所以 `btcusdt` ≠ `btc_usdt`

### 正確格式範例

#### ✅ 正確的策略配置

```python
class MyStrategy(Strategy):
    def pre_start(self, context):
        # ✓ 正確: 小寫 + 底線
        context.subscribe("binance", 
                         ["btc_usdt", "eth_usdt", "bnb_usdt"],
                         InstrumentType.Spot, 
                         Exchange.BINANCE)

    def on_depth(self, context, depth):
        # ✓ 正確: 使用相同格式
        if depth.instrument_id == "btc_usdt":
            price = depth.ask_price[0]
            
        # ✓ 正確: 下單也用相同格式
        context.insert_order(
            symbol="btc_usdt",  # ← 小寫 + 底線
            side=Side.Buy,
            price=price,
            volume=0.001,
            price_type=PriceType.Limit
        )
```

#### ❌ 錯誤的策略配置

```python
class MyStrategy(Strategy):
    def pre_start(self, context):
        # ✗ 錯誤 1: 沒有底線
        context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)
        
        # ✗ 錯誤 2: 大寫
        context.subscribe("binance", ["BTCUSDT"], InstrumentType.Spot, Exchange.BINANCE)
        
        # ✗ 錯誤 3: 使用破折號
        context.subscribe("binance", ["btc-usdt"], InstrumentType.Spot, Exchange.BINANCE)
        
        # ✗ 錯誤 4: 混合格式
        context.subscribe("binance", ["BTC_USDT"], InstrumentType.Spot, Exchange.BINANCE)

    def on_depth(self, context, depth):
        # ✗ 錯誤 5: 訂閱和下單格式不一致
        context.insert_order(
            symbol="BTCUSDT",  # ← 訂閱用 "btc_usdt",下單卻用 "BTCUSDT"
            side=Side.Buy,
            price=42000,
            volume=0.001
        )
```

### 常見錯誤診斷

#### 問題 1: 策略無法收到市場數據

**症狀**:
- `pm2 logs` 顯示策略正常啟動
- `on_depth()` 回調從未被呼叫
- MD Gateway 日誌顯示正常接收數據

**原因**: Symbol 格式錯誤,導致訂閱匹配失敗

**檢查方法**:
```bash
# 查看策略訂閱的 symbol
docker exec godzilla-dev pm2 logs my_strategy | grep "subscribe"

# 查看 MD 接收的 symbol
docker exec godzilla-dev pm2 logs md_binance | grep "symbol"
```

**解決方案**: 確保使用 `小寫_小寫` 格式

#### 問題 2: 下單時 IndexError

**症狀**:
```
IndexError: list index out of range
  File "kungfu/wingchun/book/book.py", line 123, in on_order_input
    quote_coin = splited[1]
```

**原因**: Symbol 沒有底線,`split("_")` 只返回 1 個元素

**檢查方法**:
```python
# 在策略中加入日誌
def on_depth(self, context, depth):
    context.log().info(f"Symbol: {depth.instrument_id}")  # 檢查格式
```

**解決方案**: 修改 symbol 為 `btc_usdt` 格式後**重新編譯**

**⚠️ 重要**: Symbol 在策略初始化時寫入 C++,修改後必須:
```bash
docker exec -it godzilla-dev bash -c "cd /app/core/build && make -j$(nproc)"
docker exec godzilla-dev pm2 restart my_strategy
```

### 格式轉換參考

| 交易所原格式 | Godzilla 格式 | 轉換規則 |
|------------|--------------|---------|
| `BTCUSDT` (Binance) | `btc_usdt` | 小寫 + 插入底線 |
| `BTC/USDT` (某些交易所) | `btc_usdt` | 小寫 + `/` → `_` |
| `BTC-USDT` (某些交易所) | `btc_usdt` | 小寫 + `-` → `_` |
| `btcusdt` (已經小寫) | `btc_usdt` | **手動插入底線** |

**注意**: 系統**不會自動**插入底線,必須手動確保格式正確!

### 快速參考表

| 你在做什麼 | 正確格式 | 錯誤格式 |
|-----------|---------|---------|
| ✅ 訂閱市場數據 | `["btc_usdt", "eth_usdt"]` | `["BTCUSDT"]`, `["btcusdt"]` |
| ✅ 下單 | `symbol="btc_usdt"` | `symbol="BTCUSDT"` |
| ✅ 比較 symbol | `if depth.instrument_id == "btc_usdt"` | `if depth.instrument_id == "btcusdt"` |
| ✅ 日誌輸出 | `log(f"Trading {symbol}")` (任意格式) | N/A (日誌不影響邏輯) |

---

## 相關文檔

- [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) - 配置檔案完整參考
- [../operations/debugging_guide.md](../operations/debugging_guide.md) - 除錯診斷流程
- [../contracts/strategy_context_api.md](../contracts/strategy_context_api.md) - Context API 完整參考

---

**更新時間**: 2025-12-01  
**預估 Token**: ~4500
