# MD Gateway 無連線問題修復計劃

## 📚 背景知識: 為什麼使用 Boost.ASIO?

### ASIO 是什麼?

**Boost.ASIO** = **A**synchronous **I**/**O** Library

- **用途**: 跨平台的 C++ 異步 I/O 庫
- **核心功能**: 處理網絡通信、定時器、文件 I/O 等異步操作
- **在本系統中的角色**: 管理 Binance WebSocket 連線的異步事件循環

### 為什麼 MD Gateway 需要 ASIO?

**MD Gateway 的工作流程**:

```
1. 連線到 Binance WebSocket 服務器
   ├─ wss://stream.binance.com:9443 (Spot)
   ├─ wss://fstream.binance.com/ws (Futures)
   └─ wss://dstream.binance.com/ws (Delivery)

2. 訂閱市場數據流
   ├─ Depth (orderbook updates)
   ├─ Trade (real-time trades)
   └─ Ticker (24hr statistics)

3. 持續接收數據 (異步事件驅動)
   ├─ WebSocket 消息到達 → 觸發回調
   ├─ 解析 JSON → 轉換為內部數據結構
   └─ 發送到 Wingchun 事件總線

4. 維持連線心跳 (每 30 秒)
   └─ 防止連線超時斷開
```

**為什麼需要異步?**

- WebSocket 是 **長連線**,數據隨時可能到達
- 不能用同步阻塞方式 (會卡死主線程)
- 需要事件循環 (**event loop**) 持續監聽網絡事件

### ASIO 在 MarketDataBinance 中的實現

**文件**: `core/extensions/binance/src/marketdata_binance.cpp`

```cpp
class MarketDataBinance {
private:
    boost::asio::io_context ioctx_;        // ASIO 事件循環上下文
    std::shared_ptr<std::thread> task_thread_;  // 專門運行事件循環的線程

    // WebSocket 客戶端 (基於 ASIO)
    std::shared_ptr<binapi::ws::websockets> ws_ptr_;   // Spot
    std::shared_ptr<binapi::ws::websockets> fws_ptr_;  // Futures
    std::shared_ptr<binapi::ws::websockets> dws_ptr_;  // Delivery
};

void MarketDataBinance::on_start() {
    // 1. 創建 ASIO 工作線程
    task_thread_ = std::make_shared<std::thread>([this] {
        ioctx_.run();  // ← 這會一直循環,直到 stop() 被調用!
    });

    // 2. 建立 WebSocket 連線
    ws_ptr_ = std::make_shared<binapi::ws::websockets>(
        ioctx_,  // ← WebSocket 使用 ioctx_ 處理異步事件
        "stream.binance.com",
        "9443"
    );

    // 3. 訂閱數據流
    ws_ptr_->depth(..., [this](const auto& msg) {
        // 收到 Depth 消息時的回調
        this->on_market_data(msg);
    });
}
```

**關鍵點**:
- `ioctx_.run()` 是一個 **無限循環**,持續監聽事件
- 線程會一直執行,直到顯式調用 `ioctx_.stop()`
- WebSocket 的所有異步操作都由 `ioctx_` 管理

---

## 🔍 問題診斷

### 症狀
- MD Gateway 日誌顯示 `connections: 0`
- 錯誤日誌出現 `pure virtual method called` 和 `bus error`
- 進程在 PM2 中顯示 `online` 但實際無法建立 WebSocket 連線
- 系統無市場數據進入,整個交易管道停止

### 為什麼會出現 ASIO 線程競態?

#### 問題場景重現

**正常啟動流程**:
```
1. MarketDataBinance 構造
2. on_start() 被調用
   ├─ 創建 task_thread_ 並啟動 ioctx_.run()
   ├─ 建立 WebSocket 連線
   └─ 開始接收數據

3. 線程持續運行,處理網絡事件
```

**當系統關閉或重啟時** (問題發生):
```
1. PM2 發送停止信號
2. MarketDataBinance 對象開始銷毀
3. ~MarketDataBinance() 被調用
   └─ ❌ 空實現! 什麼都不做

4. C++ 自動銷毀成員變數 (按聲明逆序):
   ├─ rest_ptr_.~shared_ptr()      ✅ 正常
   ├─ frest_ptr_.~shared_ptr()     ✅ 正常
   ├─ task_thread_.~shared_ptr()   ⚠️ 問題!
   │   └─ 線程還在運行 ioctx_.run()
   │   └─ 銷毀 shared_ptr 不會等待線程終止
   │
   ├─ dws_ptr_.~shared_ptr()       ⚠️ WebSocket 開始銷毀
   ├─ fws_ptr_.~shared_ptr()       ⚠️ 但 ASIO 線程還在訪問它們!
   ├─ ws_ptr_.~shared_ptr()        ⚠️
   │
   └─ ioctx_.~io_context()         ❌ 災難!
       └─ ASIO 線程正在執行 ioctx_.run()
       └─ 現在 ioctx_ 被銷毀了
       └─ 線程訪問已銷毀對象 → bus error

5. ASIO 線程試圖調用虛函數
   └─ 虛函數表已被銷毀
   └─ pure virtual method called
   └─ 進程崩潰
```

#### 為什麼 DEBUG 模式沒問題,RELEASE 模式會崩潰?

**DEBUG 模式 (-O1)**:
```cpp
// 編譯器生成的代碼 (保守)
~MarketDataBinance() {
    // 隱式添加額外的檢查和延遲
    // 成員銷毀順序嚴格
    // 虛函數表保留更久
}
```

**RELEASE 模式 (-O3)**:
```cpp
// 編譯器生成的代碼 (激進優化)
~MarketDataBinance() {
    // 移除冗餘檢查
    // 重排銷毀順序以提高效率
    // 內聯虛函數調用
    // 更早釋放虛函數表

    // 結果: 競態更容易觸發!
}
```

**具體差異**:

| 方面 | DEBUG (-O1) | RELEASE (-O3) |
|------|------------|---------------|
| 銷毀順序 | 嚴格按聲明逆序 | 可能重排優化 |
| 虛函數表 | 保留到最後 | 提前釋放 |
| 內存訪問 | 有邊界檢查 | 無檢查,直接訪問 |
| 時序 | 較慢,給線程更多時間 | 極快,線程來不及反應 |

**為什麼現在才出現?**

```
之前: DEBUG 模式編譯
  ↓
關閉 DEBUG → RELEASE 模式編譯
  ↓
編譯器激進優化
  ↓
隱藏的競態被觸發
  ↓
MD Gateway 崩潰
```

---

### 根本原因

**MarketDataBinance 類的銷毀實現不完整**

**文件**: `/home/huyifan/projects/godzilla-evan/core/extensions/binance/src/marketdata_binance.cpp:58`

```cpp
MarketDataBinance::~MarketDataBinance() {}  // ❌ 空實現!
```

**問題分析**:

1. **ASIO 線程未停止**:
   - `task_thread_` 在 `on_start()` 中啟動,執行 `ioctx_.run()` 無限循環
   - 銷毀時 `ioctx_` 從未調用 `stop()` 來中止循環
   - 線程繼續運行並訪問已銷毀的對象

2. **競態條件**:
   ```
   銷毀順序:
   1. ~MarketDataBinance() 執行 (空實現,什麼都不做)
   2. 自動銷毀成員變數:
      - ioctx_ 銷毀 (但線程仍在 ioctx_.run())
      - task_thread_ 銷毀 (試圖等待已死線程)
   3. ASIO 線程訪問已銷毀對象
   4. 虛函數表損壞 → pure virtual method called
   5. bus error / SIGSEGV
   ```

3. **DEBUG vs RELEASE 模式差異**:
   - **DEBUG (-O1)**: 編譯器保守,競態問題被隱藏
   - **RELEASE (-O3)**: 激進優化,曝露隱藏的缺陷
     - 對象銷毀順序優化
     - 虛函數調用內聯化
     - 內存重排

### 時間線

```
關閉 DEBUG 模式 → 重新編譯為 RELEASE (-O3)
    ↓
啟動 MD Gateway
    ↓
ASIO 線程持續運行
    ↓
進程清理觸發銷毀
    ↓
競態: 線程訪問已銷毀對象
    ↓
bus error / pure virtual called
    ↓
connections: 0 (無法建立連線)
```

---

## 🕵️ Git 歷史調查

### ASIO 的來源

**關鍵發現**:

1. **ASIO 是 upstream 原生實現** ✅
   ```bash
   git log --all --oneline extensions/binance/src/marketdata_binance.cpp
   # 結果: d11fa2e v2.1.0 (初始commit)
   ```

   - `MarketDataBinance::~MarketDataBinance() {}` **從一開始就是空實現**
   - 來自 upstream: `https://github.com/godzilla-foundation/godzilla-community.git`
   - **不是我們後來添加的,是原始代碼就有的 bug**

2. **為什麼以前沒問題?**

   **時間線分析**:
   ```
   v2.1.0 (d11fa2e)
     ↓ 空的銷毀函數一直存在
   Phase 1-4: DEBUG 模式開發
     ↓ -O1 優化,問題被隱藏
   Phase 5: 引入 latency monitoring
     ↓ 仍然 DEBUG 模式
   Phase 6 (最近):
     ↓ 關閉 DEBUG 模式 → RELEASE 模式 (-O3)
     ↓ 激進優化曝露了競態
     ↓ MD Gateway 開始崩潰 ❌
   ```

   **具體原因**:
   - **Phase 1-5**: 所有開發都在 DEBUG 模式 (`-O1 + ASAN`)
   - **Phase 6**: 為了生產部署,關閉 DEBUG → RELEASE 模式 (`-O3`)
   - **編譯器優化差異**:
     - DEBUG: 保守優化,隱藏了線程競態
     - RELEASE: 激進優化,曝露了隱藏的 bug

3. **這是一個 upstream 的長期潛在 bug**

   - 在 DEBUG 模式下不容易觸發
   - 在 RELEASE 模式下必然觸發
   - 所有使用 godzilla-community 的項目都可能受影響

### 回滾策略

#### 選項 A: 回到最後一個可運行的版本

**最後已知正常版本**:
```bash
git log --oneline | head -20
# c3a22fa refactor(strategy): migrate trading logic from on_depth to on_factor
# 79d7407 chore(phase-6): finalize production mode configuration  ← 這裡關閉了 DEBUG
# 0d07fa7 chore(hf-live): update submodule to b9d6b79
```

**回滾命令** (如果需要):
```bash
# 1. 回到 Phase 5 最後一個穩定版本
git checkout v0.5.1-phase5d-latency-monitoring

# 2. 或者只回滾 CMakeLists.txt 的 DEBUG 設置
git checkout 79d7407^ -- hf-live/CMakeLists.txt  # 恢復 DEBUG 模式

# 3. 重新編譯
docker exec godzilla-dev bash -c "cd /app/hf-live/build && cmake .. && make -j4"

# 4. 重啟服務
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh stop && ./run.sh start"
```

**回滾風險評估**:
- ✅ **安全**: 回到已知可運行狀態
- ⚠️ **代價**: 失去 Phase 6 的所有功能 (Factor 12 修復、完整市場數據)
- ⚠️ **臨時方案**: 只是規避問題,沒有真正修復

#### 選項 B: 暫時恢復 DEBUG 模式

**快速修復** (保留 Phase 6 功能):
```bash
# 修改 CMakeLists.txt
sed -i 's/option(DEBUG_MODE .* OFF)/option(DEBUG_MODE "..." ON)/' hf-live/CMakeLists.txt
sed -i 's/option(DEBUG_FACTOR_VALUES .* OFF)/option(DEBUG_FACTOR_VALUES "..." ON)/' hf-live/CMakeLists.txt

# 重新編譯
docker exec godzilla-dev bash -c "cd /app/hf-live/build && cmake .. && make -j4"
```

**優點**:
- ✅ 保留所有 Phase 6 功能
- ✅ MD Gateway 可以正常運行
- ✅ 快速恢復,風險極低

**缺點**:
- ⚠️ 性能稍差 (但仍可接受)
- ⚠️ 有大量 DEBUG 日誌 (可以關閉 DEBUG_FACTOR_VALUES 保留 DEBUG_MODE)
- ⚠️ 不是長期解決方案

#### 選項 C: 修復銷毀函數 (推薦)

**永久修復**:
- ✅ 徹底解決問題
- ✅ 可以安全使用 RELEASE 模式
- ✅ 對 upstream 有貢獻價值

**修復代碼** (已在計劃中詳述):
```cpp
MarketDataBinance::~MarketDataBinance() {
    ioctx_.stop();
    if (task_thread_ && task_thread_->joinable()) {
        task_thread_->join();
    }
    ws_ptr_.reset();
    fws_ptr_.reset();
    dws_ptr_.reset();
    rest_ptr_.reset();
    frest_ptr_.reset();
}
```

### 建議的安全流程

1. **立即**: 使用選項 B (恢復 DEBUG 模式) 讓系統運行
2. **驗證**: 確認 MD Gateway 連線正常,系統可用
3. **準備**: 在開發分支測試選項 C 的修復
4. **測試**: 在 DEBUG 和 RELEASE 模式都驗證修復
5. **部署**: 確認穩定後再切換到 RELEASE 模式

### Git 安全操作

**創建安全點**:
```bash
# 1. 創建當前狀態的 branch
git branch backup-before-md-fix

# 2. 創建修復的新 branch
git checkout -b fix/md-gateway-destructor

# 3. 進行修復...

# 4. 如果出問題,隨時可以回退
git checkout backup-before-md-fix
```

---

## 🎯 修復方案

### 方案: 實現完整的銷毀邏輯

**文件**: `/home/huyifan/projects/godzilla-evan/core/extensions/binance/src/marketdata_binance.cpp`

**位置**: 行 58

**當前代碼**:
```cpp
MarketDataBinance::~MarketDataBinance() {}
```

**修復後代碼**:
```cpp
MarketDataBinance::~MarketDataBinance() {
    // 1. 停止 ASIO 事件循環
    //    這會導致 ioctx_.run() 返回,線程可以正常終止
    ioctx_.stop();

    // 2. 等待任務線程終止
    //    確保線程完全結束後才銷毀對象
    if (task_thread_ && task_thread_->joinable()) {
        task_thread_->join();
    }

    // 3. 顯式重置 WebSocket 指針 (觸發底層清理)
    //    雖然 shared_ptr 會自動銷毀,但顯式重置可以確保順序
    ws_ptr_.reset();
    fws_ptr_.reset();
    dws_ptr_.reset();

    // 4. 重置 REST API 指針
    rest_ptr_.reset();
    frest_ptr_.reset();
}
```

**關鍵改變**:

1. **`ioctx_.stop()`**: 停止事件循環,允許 `run()` 返回
2. **`task_thread_->join()`**: 阻塞等待線程終止,避免訪問已銷毀對象
3. **顯式 `reset()`**: 確保資源按正確順序清理 (雖然不是嚴格必要)

---

## 📋 實施步驟

### Step 1: 修改 marketdata_binance.cpp
- **文件**: `/home/huyifan/projects/godzilla-evan/core/extensions/binance/src/marketdata_binance.cpp`
- **位置**: 行 58
- **操作**: 將空的銷毀函數替換為完整的清理邏輯

### Step 2: 重新編譯 Binance 擴展
```bash
docker exec godzilla-dev bash -c "cd /app/core && mkdir -p build && cd build && cmake .. && make -j4"
```

**預期輸出**:
- ✅ 編譯成功,無錯誤
- ✅ 生成新的 `kfext_binance.cpython-38-x86_64-linux-gnu.so`

### Step 3: 重啟服務
```bash
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh stop"
docker exec godzilla-dev bash -c "sleep 2 && cd /app/scripts/binance_test && ./run.sh start"
```

### Step 4: 驗證修復

#### 4.1 檢查 MD Gateway 日誌
```bash
docker exec godzilla-dev bash -c "tail -50 /home/huyifan/projects/godzilla-evan/runtime/md/binance/binance/log/live/binance.log"
```

**預期結果**:
- ✅ 無 `bus error`
- ✅ 無 `pure virtual method called`
- ✅ 看到 WebSocket 連線建立日誌

#### 4.2 檢查連線數
```bash
docker exec godzilla-dev pm2 logs md_binance --nostream --lines 50 | grep "connections:"
```

**預期結果**:
- ✅ `connections: 3` (ws_ptr, fws_ptr, dws_ptr)
- ✅ 不再是 `connections: 0`

#### 4.3 檢查進程穩定性
```bash
docker exec godzilla-dev pm2 list
```

**預期結果**:
- ✅ md_binance status = `online`
- ✅ restart count 不會持續增加
- ✅ uptime 穩定增長

#### 4.4 驗證市場數據流
```bash
docker exec godzilla-dev bash -c "tail -100 /root/.pm2/logs/strategy-test-hf-live-error.log | grep 'OnDepth\|OnTicker\|OnTrade'"
```

**預期結果**:
- ✅ 看到 `[OnDepth]`, `[OnTicker]`, `[OnTrade]` 日誌
- ✅ 市場數據正常流入策略

#### 4.5 驗證因子計算
```bash
docker exec godzilla-dev bash -c "tail -100 /root/.pm2/logs/strategy-test-hf-live-error.log | grep 'FACTOR'"
```

**預期結果**:
- ✅ 看到 `[FACTOR] 🎊 Received factor`
- ✅ Factor 12 有非零值 (ticker_momentum 修復已生效)

---

## 🔬 驗證清單

| 檢查項 | 預期結果 | 狀態 |
|--------|---------|------|
| MD Gateway 無 bus error | ✅ 無錯誤日誌 | ⬜ |
| WebSocket 連線數 | ✅ connections: 3 | ⬜ |
| 進程不崩潰 | ✅ restart count 穩定 | ⬜ |
| 市場數據流入 | ✅ OnDepth/OnTicker/OnTrade | ⬜ |
| 因子計算正常 | ✅ FACTOR_VALUES 輸出 | ⬜ |
| Factor 12 修復生效 | ✅ 非零值 | ⬜ |

---

## 📊 影響範圍

### 受影響
- ✅ MD Gateway (直接修復)
- ✅ 市場數據管道 (恢復功能)
- ✅ 策略執行 (可以接收數據)
- ✅ 因子計算 (可以正常運行)

### 不受影響
- ✅ Ledger 服務
- ✅ Master 服務
- ✅ TD Gateway
- ✅ 已保存的歷史數據

---

## ⚠️ 注意事項

### 1. 為什麼之前沒有發現?
- DEBUG 模式 (-O1) 的編譯器優化較保守,隱藏了競態
- RELEASE 模式 (-O3) 的激進優化曝露了缺陷

### 2. 是否影響其他組件?
- 需要檢查其他使用 ASIO 的組件是否有類似問題
- 建議 code review 其他 Gateway 的銷毀實現

### 3. 長期改進
- 考慮使用 RAII 包裝管理 ASIO 生命週期
- 添加單元測試驗證銷毀順序
- 在 DEBUG 和 RELEASE 模式都進行測試

---

## 📝 相關文檔更新

修復後需要更新:
- `.doc/modules/binance_extension.md` - 記錄銷毀邏輯的重要性
- `.doc/troubleshooting/md_gateway_issues.md` - 添加此問題的診斷步驟

---

## 🎯 總結

**問題**: MarketDataBinance 空銷毀實現 + RELEASE 模式編譯優化 → ASIO 線程競態

**修復**: 實現完整的銷毀邏輯 (stop ASIO → join thread → reset pointers)

**時間**: ~30 分鐘 (修改 + 編譯 + 測試)

**優先級**: 🔴 關鍵 (系統無法運行)

**驗證**: 檢查 connections 數量、進程穩定性、市場數據流
