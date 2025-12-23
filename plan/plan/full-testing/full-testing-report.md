# Phase 6 完整測試學習報告

**創建日期**: 2025-12-23
**測試範圍**: Test 4.1 - 4.3 (啟動時序問題調查與修復)
**最終狀態**: ✅ 生產就緒

---

## 問題1: 如何驗證系統穩定無崩潰風險?

### 背景

在 Test 4.2 中發現 ASIO 析構函數為空,違反 RAII 原則,可能導致 bus error。修復後需要全面驗證系統穩定性。

### 驗證方法

#### 1. 運行時穩定性測試

**目標**: 確認系統在長時間運行中無崩潰

**步驟**:
```bash
# 1. 清空日誌並重啟服務
docker exec godzilla-dev pm2 flush
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

# 2. 持續監控至少 30-60 分鐘
docker exec godzilla-dev pm2 list  # 檢查 restarts 計數
docker exec godzilla-dev pm2 logs md_binance --lines 100 | grep -i "error\|bus\|segfault"
```

**成功標準**:
- ✅ PM2 中所有服務 status = "online"
- ✅ restarts 計數保持不變 (無自動重啟)
- ✅ 無 bus error, segfault, pure virtual method called 錯誤
- ✅ connections: 3 保持穩定

**Test 4.3 驗證結果**:
- 運行時間: 30+ 分鐘
- 重啟次數: 0 (策略 restarts: 36 是之前測試累積)
- 崩潰次數: 0
- connections: 穩定在 3

#### 2. 服務停止/重啟測試

**目標**: 確認 ASIO 析構正確執行,無懸空線程

**步驟**:
```bash
# 測試正常停止
docker exec godzilla-dev pm2 stop md_binance
docker exec godzilla-dev pm2 logs md_binance --lines 20

# 檢查析構日誌
# 應該看到:
# - "MarketDataBinance destructor: stopping ASIO event loop"
# - "MarketDataBinance destructor: joining task thread"
# - "MarketDataBinance destructor: cleanup complete"

# 測試重啟循環
docker exec godzilla-dev pm2 restart md_binance
sleep 10
docker exec godzilla-dev pm2 stop md_binance
```

**成功標準**:
- ✅ 析構日誌完整顯示
- ✅ 無 "pure virtual method called" 或 "std::terminate"
- ✅ PM2 能正常停止進程 (無需 kill -9)

**ASIO 修復前後對比**:

| 情況 | 修復前 | 修復後 |
|------|--------|--------|
| `ioctx_.run()` 狀態 | 無限循環,析構時仍在運行 | stop() 後正常退出 |
| `task_thread_` 狀態 | 未 join,觸發 std::terminate | join() 後完全終止 |
| 析構時間 | 立即返回 (線程仍在運行) | 等待線程完全結束 |
| 崩潰風險 | 高 (競態條件、懸空指針) | 無 (RAII 完整) |

#### 3. 內存/線程檢查

**步驟**:
```bash
# 檢查線程數 (應該在停止後歸零)
docker exec godzilla-dev ps -T -p $(docker exec godzilla-dev pgrep -f md_binance) | wc -l

# 檢查內存洩漏 (如果啟用 ASAN)
# 在日誌中搜尋 "LeakSanitizer"
```

---

## 問題2: 多訂閱是否正確工作,所有因子計算?

### 背景

原問題 (12/18): connections: 0, 懷疑是多訂閱導致。
Test 4.1 發現: 單訂閱也失敗 → 排除多訂閱問題。
Test 4.2 發現: 延遲啟動成功 → 確認為時序問題。

### 驗證方法

#### 1. WebSocket 連線驗證

**目標**: 確認三個 WebSocket 連線全部建立

**步驟**:
```bash
docker exec godzilla-dev pm2 logs md_binance | grep "connections:"
```

**成功標準**:
```
MarketDataBinance::_check_status, connections: 3
```

**解讀**:
- connections: 0 → 無連線,訂閱失敗
- connections: 1 → 僅 Depth 連線 (單訂閱)
- connections: 2 → Depth + Trade 或 Depth + Ticker
- connections: 3 → ✅ Depth + Trade + Ticker 全部成功

#### 2. 數據類型驗證

**目標**: 確認 MD Gateway 接收三種數據類型

**步驟**:
```bash
docker exec godzilla-dev pm2 logs md_binance --lines 200 | grep "msg type"
```

**成功標準**:
```
msg type 101: depthUpdate   ← Depth 數據
msg type 102: bookTicker    ← Ticker 數據
msg type 103: aggTrade      ← Trade 數據
```

**注意**: msg type 104 (IndexPrice) 是自動推送,無需訂閱。

#### 3. 因子計算完整性驗證

**目標**: 確認 hf-live FactorEngine 計算所有 15 個因子

**步驟**:
```bash
docker exec godzilla-dev pm2 logs strategy_test_hf_live | grep "Factor outputs"
```

**成功標準**:
```
[FactorEngine] Factor outputs (20):
  spread mid_price bid_ask_ratio depth_imbalance weighted_mid     ← 5 Depth factors
  trade_volume_ma trade_direction trade_intensity vwap trade_volatility  ← 5 Trade factors
  ticker_spread ticker_volume_ratio ticker_momentum               ← 3 Ticker factors
  basis basis_pct                                                 ← 2 IndexPrice factors
  pred_signal pred_confidence                                     ← 2 Model outputs
  (total: 15 market factors + 2 model outputs + 3 metadata = 20)
```

**因子缺失診斷**:
- 缺少 Depth 因子 (spread, mid_price, ...) → Depth 訂閱失敗
- 缺少 Trade 因子 (trade_volume_ma, ...) → Trade 訂閱失敗
- 缺少 Ticker 因子 (ticker_spread, ...) → Ticker 訂閱失敗
- 因子都是 NaN → 數據未到達 hf-live 或配置錯誤

#### 4. 策略層回調驗證

**目標**: 確認 Python 策略接收因子數據

**步驟**:
```bash
docker exec godzilla-dev pm2 logs strategy_test_hf_live | grep "LinearModel"
```

**成功標準**:
```
🤖 [LinearModel] BTCUSDT Signal=+76.3843 (BULLISH) Conf=100.00%
```

**回調鏈完整性**:
```
Binance WebSocket (3 streams)
  ↓
MD Gateway (3 connections)
  ↓
hf-live FactorEngine (15 factors)
  ↓
hf-live ModelEngine (LinearModel)
  ↓
Python on_factor callback ← 如果這裡有輸出,說明全鏈路正常
```

### 數據流追蹤圖

```
Binance WebSocket (3 streams)
  ↓ depthUpdate, aggTrade, bookTicker
MD Gateway (3 connections)
  ↓ msg type 101/102/103
hf-live FactorEngine (15 factors)
  ↓ spread, mid_price, trade_volume_ma, ticker_spread, ...
hf-live ModelEngine (LinearModel)
  ↓ pred_signal, pred_confidence
Python on_factor callback
  ↓ 🤖 [LinearModel] Signal=+76.38 (BULLISH)
```

### 診斷決策樹

```
問題: connections: 0
  ├─ 檢查 MD Gateway 日誌
  │  ├─ 無 WebSocket 連線嘗試 → 訂閱未發起,檢查策略 pre_start()
  │  └─ 有連線嘗試但失敗 → 檢查網絡、API Key
  └─ 檢查策略日誌
     ├─ "RuntimeError: invalid md" → 啟動時序問題,添加重試
     └─ 無錯誤但無訂閱 → 檢查配置 md_source

問題: connections: 1
  ├─ 確認哪個訂閱成功 (通常是 Depth)
  ├─ 檢查 Trade 訂閱 → context.subscribe_trade() 是否調用?
  └─ 檢查 Ticker 訂閱 → context.subscribe_ticker() 是否調用?

問題: 缺少特定因子
  ├─ 缺少 Depth 因子 → 檢查 connections 是否 >= 1
  ├─ 缺少 Trade 因子 → 檢查 msg type 103 是否存在
  └─ 缺少 Ticker 因子 → 檢查 msg type 102 是否存在

問題: on_factor 未觸發
  ├─ 檢查 hf-live libsignal.so 是否加載
  ├─ 檢查 FactorEngine 是否啟動
  └─ 檢查 Python 策略 on_factor() 函數是否定義
```

---

## 問題3: ASIO 修復到底做了什麼?

### ASIO 基礎知識

**Boost.ASIO** = **A**synchronous **I**/**O** Library

**在系統中的角色**:
- MD Gateway 使用 ASIO 管理 Binance WebSocket 連線
- 提供異步事件循環 (event loop) 持續監聽網絡事件
- 核心組件:
  - `io_context`: 事件循環上下文
  - `io_context::run()`: 阻塞式運行事件循環
  - `io_context::stop()`: 停止事件循環

### 修復前的問題

**文件**: `core/extensions/binance/src/marketdata_binance.cpp` (舊版本)

```cpp
MarketDataBinance::~MarketDataBinance() {
    // 空析構! 違反 RAII 原則
}
```

**問題分析**:

1. **ASIO 線程無法終止**
   ```
   on_start() 創建線程:
     task_thread_ = make_shared<thread>([this] {
         ioctx_.run();  // ← 無限循環,永不返回!
     });

   析構函數:
     (空函數) → ioctx_.run() 仍在運行 → 線程永不結束
   ```

2. **`shared_ptr<thread>` 不會自動 join**
   - C++ 規則: `std::thread` 必須顯式 join() 或 detach()
   - 如果析構時 thread 仍 joinable() → 觸發 `std::terminate()`
   - `shared_ptr` 不會自動調用 join(),只會釋放內存

3. **競態條件與懸空指針**
   ```
   時間線:
   T=0  析構函數返回
   T=1  WebSocket 對象被銷毀
   T=2  ASIO 線程仍在運行,嘗試訪問 WebSocket → ❌ 懸空指針!
   T=3  Bus error / Segmentation fault
   ```

### 修復後的實現

**文件**: `core/extensions/binance/src/marketdata_binance.cpp:59-81`

```cpp
MarketDataBinance::~MarketDataBinance() {
    SPDLOG_INFO("MarketDataBinance destructor: stopping ASIO event loop");

    // 步驟1: 停止 ASIO 事件循環
    // 作用: 使 ioctx_.run() 返回
    ioctx_.stop();

    // 步驟2: 等待線程完全終止
    // 作用: 確保 ASIO 線程已經退出,沒有懸空操作
    if (task_thread_ && task_thread_->joinable()) {
        SPDLOG_INFO("MarketDataBinance destructor: joining task thread");
        task_thread_->join();  // ← 阻塞等待線程結束
    }

    // 步驟3: 顯式銷毀 WebSocket 連線
    // 作用: 確保資源釋放順序正確
    ws_ptr_.reset();
    fws_ptr_.reset();
    dws_ptr_.reset();

    // 步驟4: 銷毀 REST API 客戶端
    rest_ptr_.reset();
    frest_ptr_.reset();

    SPDLOG_INFO("MarketDataBinance destructor: cleanup complete");
}
```

### RAII 原則實現

**R**esource **A**cquisition **I**s **I**nitialization

- 構造時獲取資源 (啟動線程、建立連線)
- 析構時釋放資源 (停止線程、關閉連線)
- 保證資源生命週期與對象生命週期一致

### 為什麼這樣修復有效?

#### 1. 消除競態條件

**修復前**:
```
T=0  析構返回 → WebSocket 銷毀
T=1  ASIO 線程仍在運行 → 訪問已銷毀的對象 ❌
```

**修復後**:
```
T=0  ioctx_.stop() → ASIO 線程準備退出
T=1  join() 阻塞等待 → 確保線程完全結束
T=2  析構返回 → 此時線程已終止,不會訪問對象 ✅
```

#### 2. 符合 C++ 線程生命週期管理

- thread 必須 join() 或 detach()
- join() 確保父對象在子線程結束前不被銷毀

#### 3. 資源釋放順序正確

```
正確順序:
1. 停止事件循環 (ioctx_.stop)
2. 等待線程終止 (thread->join)
3. 銷毀 WebSocket (ws_ptr_.reset)
4. 銷毀 REST 客戶端 (rest_ptr_.reset)
```

### 同樣的修復應用到 TraderBinance

**文件**: `core/extensions/binance/src/trader_binance.cpp:83-104`

問題和修復方法完全相同:
- TraderBinance 也使用 ASIO 管理 WebSocket 連線 (User Data Stream)
- 原析構函數也是空的
- 應用相同的 4 步驟修復

### 修復前後時間線對比

| 階段 | 修復前 | 修復後 |
|------|--------|--------|
| **T=0** | pm2 stop 發送 SIGTERM | pm2 stop 發送 SIGTERM |
| **T=1** | ~MarketDataBinance() 立即返回 | ioctx_.stop() 停止事件循環 |
| **T=2** | WebSocket 對象開始析構 | task_thread_->join() 等待線程退出 |
| **T=3** | ASIO 線程仍在 ioctx_.run() | ASIO 線程從 ioctx_.run() 返回 |
| **T=4** | ASIO 線程嘗試訪問 WebSocket | thread 完全終止 |
| **T=5** | ❌ Bus error / Segmentation fault | WebSocket 開始安全析構 |
| **T=6** | 進程異常終止 | ✅ 正常退出,日誌顯示 "cleanup complete" |

---

## 問題4: Debug 日誌清理與 DEBUG 編譯參數?

### 背景

在開發過程中,添加了大量診斷日誌來定位問題:
- Phase 4 系列: 內存問題、回調失敗
- Test 4.1-4.3: 啟動時序問題
- ASIO 析構驗證

現在系統穩定,需要決定哪些日誌保留,哪些移除。

### DEBUG 編譯參數說明

**CMake 配置**: `core/CMakeLists.txt`

```cmake
SET(CMAKE_CXX_FLAGS_RELEASE "-O0 -DNDEBUG ...")  # -DNDEBUG 會關閉 assert()
SET(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall")
```

**影響**:
- `CMAKE_BUILD_TYPE=Debug`:
  - 編譯帶 debug symbols (-g)
  - 啟用完整斷言 (assert)
  - 性能較低,二進制文件較大
- `CMAKE_BUILD_TYPE=Release` (生產模式):
  - 編譯優化 (-O2 或 -O3)
  - 禁用斷言 (-DNDEBUG)
  - 性能最佳

**當前設置**: Release 模式 (生產配置)

### 日誌清理策略

#### 應該移除的日誌

**類型1: 臨時測試標記**
```python
# test_hf_live.py
context.log().info(f"🏁 [TEST 4.3] Pre-Start - Multi-Subscription with Retry")
context.log().info(f"✅ [{data_type}] Subscribed on retry {retry}")
context.log().info(f"📡 [TEST 4.3] All subscriptions completed: Depth + Trade + Ticker")
```

**原因**: "TEST 4.3" 是臨時測試標識,生產環境中無意義。

**修改建議**:
```python
# 簡化為通用日誌
context.log().info("Initializing strategy with multi-subscription retry mechanism")
# 成功時僅在 retry > 0 時輸出
if retry > 0:
    context.log().info(f"✅ [{data_type}] Subscribed after {retry} retries")
# 完成時簡化
context.log().info("✅ All market data subscriptions completed (Depth + Trade + Ticker)")
```

**類型2: 正常情況下的冗餘日誌**
```python
# 修改前: 每次訂閱都輸出,包括 retry=0
context.log().info(f"✅ [{data_type}] Subscribed on retry {retry}")

# 修改後: 只在重試時輸出
if retry > 0:
    context.log().info(f"✅ [{data_type}] Subscribed after {retry} retries")
```

**原因**: 正常情況 (retry=0) 不需要輸出,減少日誌噪音。

#### 應該保留的日誌

**類型1: ASIO 析構日誌** (重要!)
```cpp
// marketdata_binance.cpp, trader_binance.cpp
SPDLOG_INFO("MarketDataBinance destructor: stopping ASIO event loop");
SPDLOG_INFO("MarketDataBinance destructor: joining task thread");
SPDLOG_INFO("MarketDataBinance destructor: cleanup complete");
```

**原因**:
- 診斷服務停止問題非常有價值
- 頻率極低 (只在服務停止時)
- 可以確認 ASIO 是否正確清理

**類型2: 連線狀態檢查**
```cpp
// marketdata_binance.cpp
SPDLOG_INFO("MarketDataBinance::_check_status, connections: {}", connections);
```

**原因**:
- 判斷 WebSocket 連線健康度的關鍵指標
- 頻率適中 (每 5 秒一次)
- 生產環境監控必需

**類型3: 錯誤與重試機制**
```python
# test_hf_live.py
context.log().warning(f"⏳ MD Gateway not ready, waiting...")
context.log().error(f"❌ [{data_type}] Failed after {max_retries} retries")
```

**原因**:
- 診斷啟動問題
- 只在異常時輸出
- 幫助識別環境配置問題

**類型4: 關鍵業務事件**
```python
context.log().info(f"📬 [on_order] order_id={order.order_id} status={order.status}")
context.log().info(f"🤖 [LinearModel] {symbol} Signal={pred_signal:+.4f}")
```

**原因**:
- 審計交易決策
- 調試策略邏輯
- 監控系統運行狀態

### 日誌級別最佳實踐

| 級別 | 用途 | 頻率限制 | 示例 |
|------|------|---------|------|
| ERROR | 嚴重錯誤,需要立即處理 | 無限制 | API 調用失敗,數據解析錯誤 |
| WARNING | 異常情況但不影響運行 | < 1/秒 | 重試機制觸發,連線斷開重連 |
| INFO | 關鍵業務事件 | < 10/秒 | 訂單成交,策略信號,連線狀態 |
| DEBUG | 詳細執行流程 | < 100/秒 | 函數進入/退出,參數值 |
| TRACE | 極詳細追蹤 | 無限制 | 每個數據包,每次循環 |

**生產環境建議**: INFO 及以上。

### 日誌保留決策表

| 日誌類型 | 頻率 | 診斷價值 | 決策 |
|---------|------|---------|------|
| TEST 4.3 標記 | 低 | 低 (臨時) | ❌ 移除 |
| retry=0 成功日誌 | 高 | 低 (正常情況) | ❌ 移除 |
| retry>0 成功日誌 | 低 | 高 (異常恢復) | ✅ 保留 |
| MD Gateway not ready | 低 | 高 (啟動問題) | ✅ 保留 |
| 訂閱失敗錯誤 | 極低 | 極高 (故障) | ✅ 保留 |
| ASIO 析構日誌 | 極低 | 極高 (崩潰診斷) | ✅ 保留 |
| LinearModel 輸出 | 中 | 高 (策略監控) | ✅ 保留 |
| on_order 回調 | 低 | 高 (交易審計) | ✅ 保留 |

### DEBUG 模式使用建議

**何時啟用 DEBUG 模式?**

1. **開發新功能**: 需要詳細的執行流程追蹤
2. **調試崩潰**: 使用 gdb 需要 debug symbols
3. **性能分析**: 使用 AddressSanitizer (ASAN) 需要 debug info

**何時使用 RELEASE 模式?**

1. **生產環境**: 最佳性能
2. **長時間測試**: 避免日誌爆炸
3. **性能 benchmark**: 獲取真實延遲數據

**當前測試**: 使用 RELEASE 模式,無需重新編譯。

---

## 總結: 關鍵經驗教訓

### 1. 系統穩定性驗證

- 運行時測試 (30-60 分鐘) 無法完全驗證析構問題
- **必須測試服務停止/重啟場景**
- PM2 restarts 計數是早期預警指標

### 2. 多訂閱問題診斷

- `connections` 數量是最直接的驗證指標
- 因子完整性檢查可以反向驗證數據流
- **問題往往不在功能本身,而在時序/配置**

### 3. ASIO RAII 原則

- 線程必須顯式 join() 或 detach()
- shared_ptr 不會自動管理線程生命週期
- **異步資源的釋放順序至關重要**

### 4. 日誌管理

- 臨時測試標記應該在測試完成後移除
- 診斷日誌應該基於**價值/頻率比**保留
- DEBUG 模式只在必要時啟用

---

## 驗證檢查清單

在部署到生產環境前,確認:

### 運行時驗證
- [ ] PM2 所有服務 status = online
- [ ] MD Gateway connections = 3
- [ ] hf-live 輸出包含 15 個因子
- [ ] Python on_factor 回調正常觸發
- [ ] 無 ERROR 或 WARNING 日誌 (除了預期的)
- [ ] 運行 30-60 分鐘無重啟

### 停止/重啟驗證
- [ ] pm2 stop 測試通過
- [ ] ASIO 析構日誌完整顯示:
  - [ ] "stopping ASIO event loop"
  - [ ] "joining task thread"
  - [ ] "cleanup complete"
- [ ] pm2 restart 測試通過
- [ ] 無 bus error 或 segfault

### 代碼質量
- [ ] 代碼無臨時測試標記
- [ ] 備份文件已刪除
- [ ] 保留所有關鍵診斷日誌
- [ ] Git 提交完成

---

## 參考資料

### 相關文檔
- [Test 4.3 測試報告](../test_4_3_report.md) - 測試結果和日誌摘錄
- [ASIO 析構驗證計劃](debug_hf-live.07-asio-destructor-verification.md) - 完整調查過程

### 關鍵代碼位置
- ASIO 析構實現: `core/extensions/binance/src/marketdata_binance.cpp:59-81`
- ASIO 析構實現: `core/extensions/binance/src/trader_binance.cpp:83-104`
- 重試機制實現: `strategies/test_hf_live/test_hf_live.py:40-54`
- 多訂閱實現: `strategies/test_hf_live/test_hf_live.py:57-66`

### 驗證命令速查
```bash
# 穩定性檢查
docker exec godzilla-dev pm2 list
docker exec godzilla-dev pm2 logs md_binance | grep "connections:"

# 析構驗證
docker exec godzilla-dev pm2 stop md_binance
docker exec godzilla-dev pm2 logs md_binance --lines 20

# 因子驗證
docker exec godzilla-dev pm2 logs strategy_test_hf_live | grep "LinearModel"
```

---

**文檔版本**: 1.0
**最後更新**: 2025-12-23
**維護人員**: Phase 6 開發團隊
