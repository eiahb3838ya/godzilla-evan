---
title: 策略偵錯指南 (Strategy Debugging Guide)
updated_at: 2025-11-24
owner: operations-team
lang: zh-TW
tags: [debugging, troubleshooting, operations, strategy, market-data]
purpose: "系統化的策略偵錯流程與常見問題排查"
code_refs:
  - core/cpp/wingchun/src/strategy/runner.cpp:68-76
  - core/cpp/wingchun/include/kungfu/wingchun/strategy/context.h:162-171
  - core/python/kungfu/wingchun/book/book.py:122-123
---

# 策略偵錯指南

## 概覽

本文檔提供系統化的策略偵錯流程，涵蓋從服務啟動到訂單執行的完整鏈路排查。

**使用時機**:
- 策略啟動失敗
- 策略無法接收市場數據
- 下單失敗或報錯
- 訂單狀態異常
- 帳號連線問題

---

## 目錄

- [通用偵錯流程](#通用偵錯流程)
- [問題 1: 策略無法接收市場數據](#問題-1-策略無法接收市場數據)
- [問題 2: IndexError 在下單時](#問題-2-indexerror-在下單時)
- [問題 3: 策略啟動失敗](#問題-3-策略啟動失敗)
- [問題 4: 訂單狀態異常](#問題-4-訂單狀態異常)
- [問題 5: 帳號連線失敗](#問題-5-帳號連線失敗)
- [日誌分析技巧](#日誌分析技巧)
- [緊急重置流程](#緊急重置流程)

---

## 通用偵錯流程

### 1. 確認服務狀態

```bash
pm2 status

# 應該看到所有服務 online
┌─────┬──────────────────────┬─────────┬─────────┐
│ id  │ name                 │ status  │ restart │
├─────┼──────────────────────┼─────────┼─────────┤
│ 0   │ master               │ online  │ 0       │
│ 1   │ ledger               │ online  │ 0       │
│ 2   │ md:binance           │ online  │ 0       │
│ 3   │ td:binance:gz_user1  │ online  │ 0       │
│ 4   │ strategy:demo_future │ online  │ 0       │
└─────┴──────────────────────┴─────────┴─────────┘
```

**如果有服務 errored 或 stopped**:
```bash
pm2 logs <service_name> --lines 50
# 查看錯誤訊息，針對性修復
```

### 2. 檢查服務啟動順序

**正確順序** (必須依序啟動，每個服務間隔 5 秒):
1. Master (服務註冊中心)
2. Ledger (帳本與持倉追蹤)
3. MD gateway (市場數據)
4. TD gateway (交易閘道)
5. Strategy (策略)

**如果順序錯誤**:
```bash
pm2 delete all
cd scripts/binance_test
./run.sh start
```

### 3. 檢查日誌流

```bash
# 並排監控所有關鍵服務
pm2 logs master | grep ERROR &
pm2 logs md:binance | grep -E "(subscribe|BTCUSDT)" &
pm2 logs strategy:demo_future
```

---

## 問題 1: 策略無法接收市場數據

### 症狀

- ✓ 策略正常啟動
- ✓ 訂閱成功: `strategy subscribe depth from binance`
- ✓ MD gateway 接收數據: `{"s":"BTCUSDT"...}`
- ✗ `on_depth()` 永遠不被調用

### 根本原因

**訂閱匹配失敗** - C++ 運行時使用 hash 匹配訂閱，symbol 格式錯誤導致 hash 不匹配。

**程式碼位置**: [runner.cpp:68-76](../../core/cpp/wingchun/src/strategy/runner.cpp#L68-L76)
```cpp
if (context_->is_subscribed("depth", strategy.first, event->data<Depth>())) {
    strategy.second->on_depth(context_, event->data<Depth>());
}
```

如果 `is_subscribed()` 返回 false，`on_depth()` 不會被調用，且**沒有任何錯誤訊息**。

### 排查步驟

#### Step 1: 檢查 symbol 格式

```bash
cat strategies/demo_future/config.json | grep symbol

# ✓ 正確格式
"symbol": "btc_usdt"   # 小寫 + 底線

# ✗ 錯誤格式
"symbol": "btcusdt"    # 缺少底線
"symbol": "BTCUSDT"    # 大寫
"symbol": "BTC_USDT"   # 大寫 + 底線
"symbol": "btc-usdt"   # 連字符
```

**如果格式錯誤**: 修正為 `"btc_usdt"` 格式，見 [Symbol 命名規範](../40_config/NAMING_CONVENTIONS.md#二交易對命名規範)

#### Step 2: 驗證訂閱註冊

```bash
pm2 logs strategy:demo_future | grep subscribe

# ✓ 應該看到
[2025-11-24 12:00:00] strategy subscribe depth from binance [894c81dc]
```

**UID `[894c81dc]`** 是訂閱的 hash ID，如果沒看到，表示 `pre_start()` 中的 `context.subscribe()` 沒有被調用。

#### Step 3: 驗證 MD gateway 發布

```bash
pm2 logs md:binance | grep -E "(subscribe|open_frame)"

# ✓ 應該看到持續的事件發布
[2025-11-24 12:00:01] open_frame msg type 101  # 101 = Depth 事件
[2025-11-24 12:00:01] open_frame msg type 101
```

如果沒看到 `open_frame`，表示 MD gateway 沒有收到交易所數據 → 檢查 [問題 5: 帳號連線失敗](#問題-5-帳號連線失敗)

#### Step 4: 加入臨時偵錯日誌

在策略的 `on_depth()` 加入:

```python
def on_depth(context, depth):
    config = context.get_config()
    context.log().info(f"DEPTH RECEIVED - symbol: '{depth.symbol}', config: '{config['symbol']}', match: {depth.symbol == config['symbol']}")
```

**重啟策略**:
```bash
pm2 restart strategy:demo_future
```

**預期輸出**:
```
# ✓ 如果有輸出 → 訂閱成功
DEPTH RECEIVED - symbol: 'btc_usdt', config: 'btc_usdt', match: True

# ✗ 如果沒有任何輸出 → 訂閱匹配失敗 (進入 Step 5)
```

#### Step 5: 重新編譯 C++ 模組

**原因**: 即使修正了 config.json，C++ 模組可能仍有舊的編譯快取。

```bash
docker-compose exec app /bin/bash
cd /app/core/build
make -j$(nproc)

# 重啟策略
pm2 restart strategy:demo_future
```

**立即檢查日誌**:
```bash
pm2 logs strategy:demo_future

# ✓ 應該在 1-2 秒內看到
DEPTH RECEIVED - symbol: 'btc_usdt', config: 'btc_usdt', match: True
```

#### Step 6: 移除臨時日誌

確認問題解決後，移除偵錯日誌:

```python
def on_depth(context, depth):
    # context.log().info(f"DEPTH RECEIVED...")  # 移除

    # 正常策略邏輯
    if depth.bid_price[0] > 0:
        # ...
```

**重新編譯並重啟**:
```bash
cd /app/core/build
make -j$(nproc)
pm2 restart strategy:demo_future
```

### 解決方案總結

| 原因 | 修復方法 | 是否需要重新編譯 |
|------|----------|-----------------|
| Symbol 格式錯誤 | 修改 config.json 為 `"btc_usdt"` | ✓ 需要 |
| 沒有調用 subscribe | 在 `pre_start()` 加入 `context.subscribe()` | ✗ 不需要 |
| MD gateway 未啟動 | `pm2 start md:binance` | ✗ 不需要 |
| 編譯快取問題 | `cd /app/core/build && make` | ✓ (本身就是編譯) |

---

## 問題 2: IndexError 在下單時

### 症狀

```python
IndexError: list index out of range

At:
  /app/core/python/kungfu/wingchun/book/book.py(123): on_order_input
  /app/core/python/kungfu/wingchun/strategy.py(341): __insert_order
  /app/strategies/demo_future/demo_future.py(41): on_depth
```

### 根本原因

**Symbol 格式缺少底線** - `book.py` 使用 `split("_")` 提取 base/quote coin。

**程式碼位置**: [book.py:122-123](../../core/python/kungfu/wingchun/book/book.py#L122-L123)
```python
splited = input.symbol.split("_")
base_coin = splited[0]   # ✓ "btc_usdt" → "btc"
quote_coin = splited[1]  # ✗ "btcusdt" → IndexError
```

### 排查步驟

#### Step 1: 檢查錯誤堆疊

```bash
pm2 logs strategy:demo_future --lines 50 | grep -A 10 "IndexError"
```

**確認**:
- 錯誤在 `book.py(123)`: `quote_coin = splited[1]`
- 調用鏈: `on_depth` → `insert_order` → `on_order_input`

#### Step 2: 檢查策略配置

```bash
cat strategies/demo_future/config.json | jq .symbol

# ✗ 如果輸出
"btcusdt"    # 缺少底線
"BTCUSDT"    # 大寫無分隔
"btc-usdt"   # 錯誤分隔符

# ✓ 應該是
"btc_usdt"   # 小寫 + 底線
```

#### Step 3: 修正並重啟

```bash
# 修改 config.json
vim strategies/demo_future/config.json
# 改為: "symbol": "btc_usdt"

# 重新編譯 (重要!)
docker-compose exec app /bin/bash
cd /app/core/build
make -j$(nproc)

# 重啟策略
pm2 restart strategy:demo_future
```

#### Step 4: 驗證修復

```bash
pm2 logs strategy:demo_future

# ✓ 應該看到
[2025-11-24 12:01:00] on_depth called
[2025-11-24 12:01:01] order_id: 123456789

# ✗ 不應該再看到
IndexError: list index out of range
```

### 預防措施

在策略的 `pre_start()` 加入驗證:

```python
def pre_start(context):
    config = context.get_config()
    symbol = config["symbol"]

    # 驗證 symbol 格式
    if "_" not in symbol:
        raise ValueError(
            f"Invalid symbol format: '{symbol}'. "
            f"Expected format: 'base_quote' (e.g., 'btc_usdt'). "
            f"See .doc/40_config/NAMING_CONVENTIONS.md#二交易對命名規範"
        )

    if symbol != symbol.lower():
        raise ValueError(f"Symbol must be lowercase: '{symbol}' → '{symbol.lower()}'")

    # 正常訂閱邏輯
    context.subscribe(config["md_source"], [symbol], ...)
```

---

## 問題 3: 策略啟動失敗

### 症狀 A: Invalid account

```
RuntimeError: invalid account gz_user1@binance
```

**原因**: TD gateway 未啟動或 account 名稱格式錯誤。

**排查**:
```bash
# 1. 檢查 TD gateway 狀態
pm2 status td:binance:gz_user1

# 2. 檢查配置中的 account 名稱
cat strategies/demo_future/config.json | jq .account

# ✓ 應該是純帳號名稱
"gz_user1"

# ✗ 不應該加前綴
"binance_gz_user1"  # 錯誤
```

**修復**:
```bash
# 確保 TD gateway 先啟動
pm2 start td:binance:gz_user1
sleep 5

# 再啟動策略
pm2 start strategy:demo_future
```

詳見: [帳號命名機制](../40_config/NAMING_CONVENTIONS.md#一帳號命名規範)

---

### 症狀 B: Python 語法錯誤

```
SyntaxError: invalid syntax
  File "/app/strategies/demo_future/demo_future.py", line 25
```

**原因**: Python 程式碼語法錯誤。

**排查**:
```bash
# 直接執行 Python 檢查語法
docker-compose exec app python3 -m py_compile strategies/demo_future/demo_future.py

# 如果有語法錯誤會立即顯示
```

**修復**: 根據錯誤訊息修正 Python 語法，重啟策略即可 (不需要重新編譯)。

---

## 問題 4: 訂單狀態異常

### 症狀 A: Order 永遠是 Pending

```python
# on_order() 只看到 status=0 (Pending)，沒有後續狀態
```

**原因**: TD gateway 未連線到交易所。

**排查**:
```bash
pm2 logs td:binance:gz_user1 | grep -E "(login|connect|error)"

# ✓ 正常應該看到
[2025-11-24 12:00:00] login successful

# ✗ 如果看到
[2025-11-24 12:00:00] login failed: -2015 Invalid API-key
[2025-11-24 12:00:00] connection refused
```

**修復**: 見 [問題 5: 帳號連線失敗](#問題-5-帳號連線失敗)

---

### 症狀 B: Order 狀態跳到 Error

```python
# on_order() 收到 status=8 (Error)
order.error_msg = "Insufficient balance"
```

**原因**: 交易所拒絕訂單 (餘額不足、價格超出限制等)。

**排查**:
```bash
pm2 logs strategy:demo_future | grep error_msg

# 常見錯誤訊息
"Insufficient balance"         # 餘額不足
"Price exceeds limit"          # 價格超出限制
"Market closed"                # 市場關閉
"Order would immediately match" # Spot 限價單會立即成交 (Post-Only 被拒)
```

**修復**:
- 檢查帳戶餘額
- 調整訂單價格、數量
- 確認市場開盤時間

---

## 問題 5: 帳號連線失敗

### 症狀 A: TD gateway login failed

```
[td:binance:gz_user1] login failed: -2015 Invalid API-key, IP, or permissions for action
```

**排查**:

```bash
# 1. 檢查 API key 配置
docker-compose exec -T app python3 -c "
import sqlite3
conn = sqlite3.connect('/app/runtime/system/etc/kungfu/db/live/accounts.db')
cursor = conn.cursor()
cursor.execute('SELECT account_id, config FROM account_config WHERE account_id = \"binance_gz_user1\"')
row = cursor.fetchone()
import json
config = json.loads(row[1])
print(f'access_key: {config[\"access_key\"][:8]}...')
print(f'secret_key: {config[\"secret_key\"][:8]}...')
"

# 2. 檢查是否在使用 Testnet
# Testnet API key 與 Mainnet 不同
```

**修復**:
```bash
# 重新配置帳號
kfc account -s binance remove
kfc account -s binance add

# 輸入正確的 API key
? Enter user_id: gz_user1
? Enter access_key: YOUR_ACCESS_KEY
? Enter secret_key: YOUR_SECRET_KEY

# 重啟 TD gateway
pm2 restart td:binance:gz_user1
```

---

### 症狀 B: MD gateway 無數據

```bash
pm2 logs md:binance

# ✗ 沒有任何 WebSocket 訊息
# ✗ 沒有 {"s":"BTCUSDT"...}
```

**排查**:
```bash
# 檢查 WebSocket 連線
pm2 logs md:binance | grep -E "(connect|subscribe|error)"

# ✓ 正常應該看到
[2025-11-24 12:00:00] subscribe btcusdt@depth10@100ms
[2025-11-24 12:00:01] {"s":"BTCUSDT","b":[...]}

# ✗ 如果看到
[2025-11-24 12:00:00] WebSocket connection failed
[2025-11-24 12:00:00] DNS resolution failed
```

**修復**:
```bash
# 1. 檢查網路連線
docker-compose exec app ping stream.binance.com

# 2. 檢查防火牆
# Binance WebSocket: wss://stream.binance.com:9443

# 3. 重啟 MD gateway
pm2 restart md:binance
```

---

## 日誌分析技巧

### 並排監控多個服務

```bash
# 使用 tmux 分割視窗
tmux new-session \; \
  split-window -h \; \
  split-window -v \; \
  select-pane -t 0 \; \
  send-keys 'pm2 logs md:binance' C-m \; \
  select-pane -t 1 \; \
  send-keys 'pm2 logs td:binance:gz_user1' C-m \; \
  select-pane -t 2 \; \
  send-keys 'pm2 logs strategy:demo_future' C-m
```

### 過濾關鍵訊息

```bash
# 只看錯誤
pm2 logs strategy:demo_future | grep -i error

# 只看訂單事件
pm2 logs strategy:demo_future | grep -E "(order_id|on_order|on_trade)"

# 只看市場數據
pm2 logs strategy:demo_future | grep -E "(on_depth|on_ticker)"

# 排除噪音日誌
pm2 logs md:binance | grep -v "heartbeat"
```

### 時間戳對齊

```bash
# 追蹤單一訂單的完整生命週期
ORDER_ID=123456789
pm2 logs strategy:demo_future | grep $ORDER_ID

# 應該看到
[12:00:00] insert_order returned: 123456789
[12:00:01] on_order: order_id=123456789, status=Pending
[12:00:02] on_order: order_id=123456789, status=Submitted, ex_order_id=999888
[12:00:05] on_order: order_id=123456789, status=Filled, avg_price=96000.0
[12:00:05] on_trade: order_id=123456789, volume=0.005, price=96000.0
```

---

## 緊急重置流程

### 完全清空重新開始

**警告**: 這會刪除所有 journal 歷史記錄！只在開發環境使用。

```bash
# 1. 停止所有服務
pm2 stop all
pm2 delete all

# 2. 清空 journal
docker-compose exec app /bin/bash
rm -rf /app/runtime/journal/*

# 3. 清空 PM2 日誌
pm2 flush

# 4. 重新啟動
cd /app/scripts/binance_test
./run.sh start

# 5. 驗證
pm2 status
pm2 logs
```

### 只重啟策略 (保留歷史)

```bash
pm2 restart strategy:demo_future
pm2 logs strategy:demo_future --lines 0
```

### 重新編譯 + 重啟

```bash
# 1. 重新編譯
docker-compose exec app /bin/bash
cd /app/core/build
make -j$(nproc)

# 2. 重啟受影響的服務
pm2 restart md:binance
pm2 restart td:binance:gz_user1
pm2 restart strategy:demo_future

# 3. 驗證
sleep 5
pm2 logs strategy:demo_future | grep -E "(on_depth|on_order)"
```

---

## 偵錯清單 (Checklist)

啟動策略前，快速檢查:

```bash
# ✓ 服務狀態
pm2 status | grep online

# ✓ Symbol 格式
cat strategies/*/config.json | jq .symbol | grep "_"

# ✓ Account 名稱
cat strategies/*/config.json | jq .account | grep -v "binance_"

# ✓ TD gateway 已登入
pm2 logs td:binance:gz_user1 | grep "login successful"

# ✓ MD gateway 有數據
pm2 logs md:binance | tail -10 | grep "BTCUSDT"

# ✓ 最近一次編譯
stat /app/core/build/kfc/python/kungfu/wingchun/*.so | grep Modify
```

---

## Related Documentation

- [Symbol 命名規範](../40_config/NAMING_CONVENTIONS.md#二交易對命名規範) - Symbol 格式詳細說明
- [帳號命名機制](../40_config/NAMING_CONVENTIONS.md#一帳號命名規範) - Account 名稱格式
- [CLI 操作指南](cli_operations_guide.md) - 服務啟動與管理
- [PM2 啟動指南](pm2_startup_guide.md) - PM2 使用方法

---

## 常見問題 (FAQ)

**Q: 為什麼修改 Python 策略程式碼後需要重新編譯 C++？**
A: 通常**不需要**。只有修改 symbol 格式、或懷疑有編譯快取問題時才需要。

**Q: 如何確認 symbol 格式問題已解決？**
A: 重啟策略後，在 1-2 秒內應該看到 `on_depth()` 的日誌輸出。

**Q: 為什麼 MD gateway 有數據但策略收不到？**
A: 99% 是 symbol 格式問題導致訂閱 hash 不匹配。檢查 config.json 的 `symbol` 欄位。

**Q: 如何快速定位是哪個環節出問題？**
A: 按照 **Master → Ledger → MD → TD → Strategy** 順序，逐一檢查日誌，找到第一個異常的服務。

**Q: 偵錯時應該加多少日誌？**
A: 只加**關鍵決策點**的日誌 (如訂閱成功、數據到達、下單觸發)。避免在高頻回調中加日誌 (如每次 `on_depth`)，會嚴重影響性能。

---

**版本**: 2025-11-24
**維護者**: operations-team
**Token 估算**: ~6,500 tokens
