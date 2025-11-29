---
title: 交易對命名規範 (Symbol Naming Convention)
updated_at: 2025-11-24
owner: config-team
lang: zh-TW
tags: [config, symbol, naming, convention, troubleshooting]
purpose: "說明系統中交易對名稱的格式要求與常見錯誤"
code_refs:
  - core/python/kungfu/wingchun/book/book.py:122-123
  - core/cpp/wingchun/include/kungfu/wingchun/common.h:354-365
  - core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h:162-171
  - core/cpp/wingchun/src/strategy/runner.cpp:68-76
  - core/extensions/binance/include/type_convert_binance.h:111-121
---

# 交易對命名規範 (Symbol Naming Convention)

## 概覽

Godzilla 交易系統中的交易對名稱有**嚴格的格式要求**：

**✓ 正確格式**: `小寫基幣_小寫報價幣` (如 `btc_usdt`, `eth_usdt`)
**✗ 錯誤格式**: `btcusdt`, `BTCUSDT`, `BTC_USDT`, `btc-usdt`

使用錯誤的格式會導致：
1. **IndexError** 在下單時 (base/quote coin 解析失敗)
2. **靜默訂閱失敗** (策略無法收到市場數據)
3. **需要重新編譯 C++** 才能修復

---

## 為什麼格式這麼重要？

### 1. Base/Quote Coin 解析 ([book.py:122-123](../../core/python/kungfu/wingchun/book/book.py#L122-L123))

當策略下單時，系統需要從交易對中提取基幣和報價幣：

```python
def on_order_input(self, input):
    # ...
    splited = input.symbol.split("_")
    base_coin = splited[0]   # ✓ "btc_usdt" → "btc"
    quote_coin = splited[1]  # ✓ "btc_usdt" → "usdt"
    # ...
```

**❌ 錯誤範例**：
```python
# 如果 symbol = "btcusdt" (沒有底線)
splited = "btcusdt".split("_")  # → ["btcusdt"] (只有 1 個元素)
base_coin = splited[0]          # → "btcusdt" (✓ 可存取)
quote_coin = splited[1]         # → IndexError: list index out of range (✗)
```

### 2. 訂閱匹配機制 ([common.h:354-365](../../core/cpp/wingchun/include/kungfu/wingchun/common.h#L354-L365))

策略訂閱市場數據時，系統使用 **symbol hash** 來匹配事件：

```cpp
inline uint32_t get_symbol_id(const std::string &symbol, const std::string &sub_type,
                                InstrumentType type, const std::string &exchange, uint32_t index)
{
    return yijinjing::util::hash_str_32(symbol) \
        ^ yijinjing::util::hash_str_32(sub_type) \
        ^ yijinjing::util::hash_str_32(std::to_string(static_cast<int>(type))) \
        ^ yijinjing::util::hash_str_32(exchange) \
        ^ yijinjing::util::hash_str_32(std::to_string(index));
}
```

**關鍵點**：symbol 字串**直接被 hash**，因此：
- `"btc_usdt"` 的 hash ≠ `"btcusdt"` 的 hash
- 如果配置檔使用 `"btcusdt"`，但市場數據發布者使用 `"btc_usdt"`，hash 不匹配 → 事件被過濾

### 3. 訂閱過濾邏輯 ([runner.cpp:68-76](../../core/cpp/wingchun/src/strategy/runner.cpp#L68-L76))

市場數據事件到達時，C++ 運行時會檢查訂閱：

```cpp
events_ | is(msg::type::Depth) |
$([&](event_ptr event)
{
    for (const auto &strategy : strategies_)
    {
        context_->set_current_strategy_index(strategy.first);
        if (context_->is_subscribed("depth", strategy.first, event->data<Depth>())) {
            strategy.second->on_depth(context_, event->data<Depth>());
        }
    }
});
```

**第 72 行的 `is_subscribed()` 檢查**：
```cpp
// context.h:162-171
template <class T>
bool is_subscribed(const std::string &sub_type, uint32_t hash_id, const T &data)
{
    if (subscribe_all_) {
        return true;
    } else {
        auto symbol_id = get_symbol_id(data.symbol, sub_type, data.instrument_type, data.exchange_id, hash_id);
        return subscribed_symbols_.find(symbol_id) != subscribed_symbols_.end();
    }
}
```

**靜默失敗**：如果 hash 不匹配，`on_depth()` **永遠不會被調用**，且沒有任何錯誤訊息。

---

## 完整流程追蹤

### 情境 1: 正確的流程 (✓)

**1. 策略配置** (`strategies/demo_future/config.json`):
```json
{
  "name": "demo_future",
  "md_source": "binance",
  "symbol": "btc_usdt"  // ✓ 正確格式
}
```

**2. 策略訂閱** (`demo_future.py`):
```python
def pre_start(context):
    config = context.get_config()
    context.subscribe(config["md_source"], [config["symbol"]],
                      InstrumentType.Spot, Exchange.BINANCE)
    # → 註冊 hash("btc_usdt" + "depth" + ...)
```

**3. MD Gateway 發布** ([marketdata_binance.cpp:148](../../core/extensions/binance/src/marketdata_binance.cpp#L148)):
```cpp
std::string orig_symbol(inst.symbol);  // "btc_usdt" (保留原始格式)
// ...
strcpy(depth.symbol, orig_symbol.c_str());  // 發布 "btc_usdt"
```

**4. 策略接收**:
```cpp
// runner.cpp:72
if (is_subscribed("depth", strategy_id, depth)) {  // ✓ hash 匹配
    strategy.second->on_depth(context_, depth);     // ✓ 回調觸發
}
```

**結果**: ✓ 策略成功接收市場數據並下單

---

### 情境 2: 錯誤的流程 (✗)

**1. 策略配置** (`strategies/demo_future/config.json`):
```json
{
  "symbol": "btcusdt"  // ✗ 錯誤格式 (沒有底線)
}
```

**2. 策略訂閱**:
```python
context.subscribe("binance", ["btcusdt"], ...)
# → 註冊 hash("btcusdt" + "depth" + ...)
```

**3. MD Gateway 發布**:
```cpp
strcpy(depth.symbol, "btc_usdt");  // 發布 "btc_usdt" (MD gateway 使用正確格式)
```

**4. 策略接收**:
```cpp
// runner.cpp:72
if (is_subscribed("depth", strategy_id, depth)) {
    // ✗ hash("btc_usdt"...) ≠ hash("btcusdt"...)
    // ✗ 條件為 false，on_depth() 不會被調用
}
```

**5. 策略嘗試下單** (假設手動觸發):
```python
context.insert_order("btcusdt", ...)  # → book.py:122-123
# ✗ IndexError: list index out of range
```

**結果**:
- ✗ 策略無法接收市場數據 (靜默失敗)
- ✗ 下單失敗 (IndexError)

---

## 交易所格式轉換

### 內部格式 vs 交易所格式

| 層級 | 格式 | 範例 | 位置 |
|------|------|------|------|
| 策略配置 | `小寫_小寫` | `"btc_usdt"` | `strategies/*/config.json` |
| Python API | `小寫_小寫` | `"btc_usdt"` | `context.subscribe()` |
| C++ 內部 | `小寫_小寫` | `"btc_usdt"` | `depth.symbol` |
| Binance WebSocket | `大寫無分隔` | `"BTCUSDT"` | WebSocket URL |
| Binance REST API | `大寫無分隔` | `"BTCUSDT"` | REST request |

### 自動轉換機制 ([type_convert_binance.h:111-121](../../core/extensions/binance/include/type_convert_binance.h#L111-L121))

MD gateway 自動處理格式轉換：

```cpp
inline const std::string to_binance_symbol(const std::string &symbol)
{
    // "btc_usdt" → "BTCUSDT"
    std::vector<std::string> result;
    std::string res;
    boost::split(result, symbol, boost::is_any_of("_"));
    for (auto &coin: result)
    {
        std::transform(coin.begin(), coin.end(), coin.begin(),
                       [](unsigned char c){ return std::toupper(c); });
        res.append(coin);
    }
    return res;
}
```

**使用**:
```cpp
std::string symbol = to_binance_symbol("btc_usdt");  // → "BTCUSDT"
ws_->subscribe_part_depth(symbol, 10);                // WebSocket 使用 "BTCUSDT"
strcpy(depth.symbol, orig_symbol.c_str());            // 發布使用 "btc_usdt"
```

**關鍵**:
- MD gateway **對外**使用交易所格式 (`"BTCUSDT"`)
- MD gateway **對內**發布時保留原始格式 (`"btc_usdt"`)
- 策略**永遠**只看到內部格式 (`"btc_usdt"`)

---

## 常見錯誤與修復

### 錯誤 1: 使用交易所格式 (大寫無分隔)

**❌ 錯誤配置**:
```json
{
  "symbol": "BTCUSDT"
}
```

**錯誤訊息**:
```
IndexError: list index out of range
  at book.py(123): quote_coin = splited[1]
```

**修復**:
```json
{
  "symbol": "btc_usdt"  // ✓
}
```

---

### 錯誤 2: 缺少底線分隔符

**❌ 錯誤配置**:
```json
{
  "symbol": "btcusdt"
}
```

**症狀**:
- ✓ 策略正常啟動
- ✓ 訂閱成功: `strategy subscribe depth from binance`
- ✓ MD gateway 接收數據: `{"s":"BTCUSDT"...}`
- ✗ `on_depth()` 永遠不被調用 (靜默失敗)

**修復**:
```json
{
  "symbol": "btc_usdt"  // ✓
}
```

**如果修復後仍無效**:
```bash
# 需要重新編譯 C++ 模組
cd /app/core/build
make -j$(nproc)

# 重啟策略
pm2 restart strategy:demo_future
```

---

### 錯誤 3: 使用連字符分隔

**❌ 錯誤配置**:
```json
{
  "symbol": "btc-usdt"
}
```

**錯誤**:
```python
splited = "btc-usdt".split("_")  # → ["btc-usdt"] (只有 1 個元素)
base_coin = splited[0]            # → "btc-usdt" (✗ 應該是 "btc")
quote_coin = splited[1]           # → IndexError
```

**修復**:
```json
{
  "symbol": "btc_usdt"  // ✓
}
```

---

### 錯誤 4: 大寫 + 底線

**❌ 錯誤配置**:
```json
{
  "symbol": "BTC_USDT"
}
```

**問題**:
- Base/quote coin 解析會成功 (`["BTC", "USDT"]`)
- 但訂閱匹配會失敗 (hash 不匹配)

**修復**:
```json
{
  "symbol": "btc_usdt"  // ✓ 必須小寫
}
```

---

## 編譯依賴問題

### 為什麼需要重新編譯？

當你修正 symbol 格式後，如果策略**仍然無法接收市場數據**，原因是：

1. **模板編譯快取**: `is_subscribed<T>()` 是 C++ 模板函數
2. **Hash 預計算**: `get_symbol_id()` 可能在編譯時優化
3. **Inline 函數**: `to_binance_symbol()` 是 inline，可能被內聯到多處

**解決方法**:
```bash
cd /app/core/build
make -j$(nproc)
```

**何時需要重新編譯**:
- ✓ 修改配置檔中的 `symbol` 欄位後
- ✓ 修改頭檔 (`.h`, `.hpp`) 中的任何程式碼
- ✗ 只修改 Python 策略程式碼 (不需要)
- ✗ 只修改 PM2 設定 (不需要)

---

## 驗證清單

### 配置檢查

在啟動策略前，檢查 `strategies/*/config.json`:

```bash
# ✓ 正確
"symbol": "btc_usdt"   # 小寫 + 底線
"symbol": "eth_usdt"
"symbol": "sol_usdt"

# ✗ 錯誤
"symbol": "BTCUSDT"    # 大寫無分隔
"symbol": "btcusdt"    # 小寫無分隔
"symbol": "btc-usdt"   # 連字符
"symbol": "BTC_USDT"   # 大寫有分隔
```

### 訂閱驗證

啟動策略後，檢查 PM2 日誌:

```bash
pm2 logs strategy:demo_future

# ✓ 應該看到
[2025-11-24 12:00:00] strategy subscribe depth from binance [894c81dc]

# ✓ 接著應該看到 (約 1 秒內)
[2025-11-24 12:00:01] on_depth called - symbol: btc_usdt

# ✗ 如果只看到 subscribe 但沒看到 on_depth
# → 訂閱匹配失敗，檢查 symbol 格式並重新編譯
```

### MD Gateway 驗證

確認 MD gateway 有接收數據:

```bash
pm2 logs md:binance | grep BTCUSDT

# ✓ 應該看到持續的 WebSocket 訊息
{"s":"BTCUSDT","b":[["96000.00","1.5"],...]}
{"s":"BTCUSDT","b":[["96001.00","2.3"],...]}
```

如果 MD gateway 有數據但策略沒有，**99% 是 symbol 格式問題**。

---

## 最佳實踐

### ✓ 正確的策略配置流程

1. **使用範本**:
   ```bash
   cp strategies/demo_future/config.json strategies/my_strategy/config.json
   ```

2. **修改 symbol 時只改值，不改格式**:
   ```json
   {
     "symbol": "btc_usdt"  // ✓ 保持 lowercase_lowercase 格式
   }
   ```

3. **支援多個交易對**:
   ```python
   # 如果需要多個交易對，在 pre_start 中訂閱
   symbols = ["btc_usdt", "eth_usdt", "sol_usdt"]
   context.subscribe("binance", symbols, InstrumentType.Spot, Exchange.BINANCE)
   ```

4. **驗證格式**:
   ```python
   # 在策略中加入驗證
   def pre_start(context):
       config = context.get_config()
       symbol = config["symbol"]
       if "_" not in symbol:
           raise ValueError(f"Invalid symbol format: {symbol}. Expected format: 'base_quote' (e.g., 'btc_usdt')")
   ```

### ✓ 偵錯時的 symbol 格式檢查

在 `on_depth()` 中加入臨時日誌:

```python
def on_depth(context, depth):
    config = context.get_config()
    context.log().info(f"Received depth - symbol: '{depth.symbol}', config: '{config['symbol']}', match: {depth.symbol == config['symbol']}")
```

**✓ 正確輸出**:
```
Received depth - symbol: 'btc_usdt', config: 'btc_usdt', match: True
```

**✗ 錯誤輸出** (如果匹配失敗):
```
# 不會有任何輸出，因為 on_depth() 不會被調用
```

---

## 技術細節：Symbol Hash 系統

### Hash 組成

訂閱 ID 由以下元素組合 hash:

```cpp
hash(symbol) ^ hash(sub_type) ^ hash(instrument_type) ^ hash(exchange) ^ hash(index)
```

**範例**:
```cpp
// 策略訂閱 "btc_usdt"
symbol_id = hash("btc_usdt") ^ hash("depth") ^ hash("0") ^ hash("binance") ^ hash("0")
// → 0xABCD1234

// 市場數據事件 "btc_usdt"
event_id = hash("btc_usdt") ^ hash("depth") ^ hash("0") ^ hash("binance") ^ hash("0")
// → 0xABCD1234 (✓ 匹配)

// 市場數據事件 "btcusdt" (錯誤格式)
event_id = hash("btcusdt") ^ hash("depth") ^ hash("0") ^ hash("binance") ^ hash("0")
// → 0x5678DCBA (✗ 不匹配)
```

### 為何使用 Hash？

1. **性能**: O(1) 查找，不需要字串比較
2. **空間**: 32-bit hash vs 完整字串
3. **彈性**: 支援多維度匹配 (symbol + type + exchange)

### Hash 碰撞？

32-bit hash 碰撞機率極低:
- 不同交易對使用不同 symbol 字串
- 即使 hash 碰撞，其他維度 (type, exchange) 仍會區分

---

## Related Documentation

- [配置使用地圖](config_usage_map.md) - 所有配置欄位的完整映射
- [策略框架](../10_modules/strategy_framework.md) - 策略配置與 API 使用
- [偵錯指南](../90_operations/debugging_guide.md) - 策略無法接收數據的完整排查流程
- [帳號命名機制](account_naming_convention.md) - 另一個容易混淆的命名系統

---

## 常見問題 (FAQ)

**Q: 為什麼不統一使用交易所格式 (BTCUSDT)？**
A:
1. 內部需要分離 base/quote coin (`book.py` 的 `split("_")`)
2. 底線格式更易讀、易維護
3. 符合多數交易所的內部慣例 (如 FTX, OKX)

**Q: 我可以用大寫 + 底線 (BTC_USDT) 嗎？**
A: ❌ 不可以。系統預期小寫格式，大寫會導致 hash 不匹配。

**Q: 如果我需要支援新的交易所，symbol 格式會改變嗎？**
A: 不會。內部統一使用 `lowercase_lowercase` 格式，每個交易所的 extension 負責轉換。

**Q: 如何批量驗證所有策略的 symbol 格式？**
A:
```bash
find strategies -name "config.json" -exec grep -H '"symbol"' {} \; | grep -v '"symbol": "[a-z]*_[a-z]*"'
# 如果有輸出，表示有策略使用錯誤格式
```

**Q: 為什麼修改配置後需要重新編譯 C++？**
A: 模板函數 (`is_subscribed<T>`) 和 inline 函數 (`get_symbol_id`) 可能在編譯時優化。雖然不應該需要，但實務上遇到過舊編譯快取導致訂閱失敗的情況。

---

**版本**: 2025-11-24
**維護者**: config-team
**Token 估算**: ~5,800 tokens
