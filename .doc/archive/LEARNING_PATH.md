# 量化交易系統學習路徑

本文件提供**任務導向的學習路徑**,通過交叉引用現有文檔快速上手。

---

## 📖 學習路徑概覽

### 第1天: 快速啟動系統 (2小時)

**目標**: 在測試網啟動完整交易系統

**步驟**:
1. **環境準備** - 閱讀 [TESTNET.md](TESTNET.md#環境準備) 完成 Docker 安裝
2. **獲取 API Keys** - 閱讀 [TESTNET.md](TESTNET.md#獲取-api-keys) 申請測試網金鑰
3. **配置帳戶** - 閱讀 [config/config_usage_map.md](../config/config_usage_map.md) 設定 TD 配置
4. **啟動服務** - 執行 [QUICK_START.md](../operations/QUICK_START.md#啟動所有服務)
5. **驗證運行** - 確認 `docker exec godzilla-dev pm2 list` 顯示所有服務 online

**驗證成果**:
```bash
# 應看到以下服務都是 online 狀態
pm2 list
# ├─ master    - online
# ├─ ledger    - online
# ├─ md_binance - online
# └─ td_binance - online
```

---

### 第2天: 編寫第一個策略 (4小時)

**目標**: 開發並運行簡單的市場數據監控策略

**步驟**:
1. **理解策略框架** - 閱讀 [modules/strategy_framework.md](../modules/strategy_framework.md)
   - 策略生命週期 (pre_start → on_depth → pre_stop)
   - 回調函數時序

2. **學習 Context API** - 閱讀 [contracts/strategy_context_api.md](../contracts/strategy_context_api.md)
   - `add_account()` - 添加交易帳戶
   - `subscribe()` - 訂閱市場數據
   - `log()` - 日誌輸出

3. **複製範例策略** - 參考 `strategies/demo_spot.py`
   ```python
   class MyFirstStrategy(Strategy):
       def pre_start(self, context):
           context.add_account("binance", "my_test_account")
           context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)

       def on_depth(self, context, depth):
           context.log().info(f"BTC best bid: {depth.bid_price[0]}")
   ```

4. **配置 PM2** - 創建 `scripts/my_first/strategy_my_first.json`
5. **啟動策略** - `docker exec godzilla-dev pm2 start /app/scripts/my_first/strategy_my_first.json`
6. **排查問題** - 參考 [operations/debugging_guide.md](../operations/debugging_guide.md)

**驗證成果**:
```bash
# 應看到策略日誌輸出市場數據
docker exec -it godzilla-dev pm2 logs my_first_strategy
# [INFO] BTC best bid: 42350.5
```

---

### 第3天: 理解訂單與交易流程 (3小時)

**目標**: 理解訂單生命週期,實作簡單下單策略

**步驟**:
1. **訂單生命週期** - 閱讀 [modules/order_lifecycle_flow.md](../modules/order_lifecycle_flow.md)
   - 訂單狀態機 (Pending → Submitted → PartialFilled → Filled)
   - `order_id` vs `ex_order_id` 的區別

2. **Order 物件契約** - 閱讀 [contracts/order_object_contract.md](../contracts/order_object_contract.md)
   - 關鍵欄位: `status`, `volume`, `volume_traded`, `avg_price`
   - 不變量: `volume_traded ≤ volume`
   - 陷阱: `ex_order_id` 在 `status=Submitted` 後才有值

3. **下單 API** - 閱讀 [contracts/strategy_context_api.md](../contracts/strategy_context_api.md#下單操作)
   ```python
   def on_depth(self, context, depth):
       if depth.bid_price[0] < self.buy_threshold:
           order_id = context.insert_order(
               symbol="btcusdt",
               side=Side.Buy,
               price=depth.bid_price[0],
               volume=0.001,
               price_type=PriceType.Limit
           )

   def on_order(self, context, order):
       if order.status == OrderStatus.Filled:
           context.log().info(f"Order {order.order_id} filled at {order.avg_price}")
   ```

4. **測試下單** - 在測試網執行小額下單
5. **查看持倉** - 理解 [modules/ledger_system.md](../modules/ledger_system.md) 持倉追蹤

**驗證成果**:
- 策略成功下單
- 接收到 `on_order()` 回調
- 確認測試網帳戶持倉變化

---

### 第4-5天: 深入系統架構 (可選,6小時)

**目標**: 理解底層架構,為複雜策略開發打基礎

**步驟**:
1. **事件溯源機制** - 閱讀 [modules/yijinjing.md](../modules/yijinjing.md)
   - Journal 的 append-only 特性
   - Reader/Writer 模式
   - 時間旅行除錯

2. **交易引擎架構** - 閱讀 [modules/wingchun.md](../modules/wingchun.md)
   - Strategy Runner + Broker + Book + Gateway 分層
   - 事件路由機制

3. **完整事件流** - 閱讀 [modules/event_flow.md](../modules/event_flow.md)
   - MD → Yijinjing → Strategy → Yijinjing → TD 流程

4. **Binance 實作** - 閱讀 [modules/binance_extension.md](../modules/binance_extension.md)
   - REST API + WebSocket 實作細節
   - 市場切換功能 (Spot/Futures)

5. **架構決策** - 閱讀 [adr/](../adr/) 目錄
   - 為什麼用 Docker ([001-docker.md](../adr/001-docker.md))
   - 為什麼用 Journal ([modules/yijinjing.md](../modules/yijinjing.md#設計理念))

**驗證成果**:
- 能繪製完整的事件流程圖
- 理解每個模組的職責與交互
- 能閱讀 C++ 核心程式碼

---

## 🎯 按任務類型查詢

| 我想... | 閱讀文檔 | 預計時間 |
|--------|---------|---------|
| **快速啟動系統** | [TESTNET.md](TESTNET.md) → [QUICK_START.md](../operations/QUICK_START.md) | 30分鐘 |
| **開發新策略** | [strategy_framework.md](../modules/strategy_framework.md) → [strategy_context_api.md](../contracts/strategy_context_api.md) | 2小時 |
| **理解訂單流程** | [order_lifecycle_flow.md](../modules/order_lifecycle_flow.md) → [order_object_contract.md](../contracts/order_object_contract.md) | 1小時 |
| **除錯策略問題** | [debugging_guide.md](../operations/debugging_guide.md) | 30分鐘 |
| **理解整體架構** | [yijinjing.md](../modules/yijinjing.md) → [wingchun.md](../modules/wingchun.md) → [event_flow.md](../modules/event_flow.md) | 3小時 |
| **配置管理** | [config_usage_map.md](../config/config_usage_map.md) → [dangerous_keys.md](../config/dangerous_keys.md) | 30分鐘 |
| **新增交易所** | [wingchun.md](../modules/wingchun.md) → [binance_extension.md](../modules/binance_extension.md) | 6小時 |

---

## ❓ 常見問題快速跳轉

### 問題: 策略無法接收市場數據

**排查步驟**: [debugging_guide.md - 問題1](../operations/debugging_guide.md#問題-1-策略無法接收市場數據)

**常見原因**:
1. Symbol 格式錯誤 → 參考 [symbol_naming_convention.md](../config/symbol_naming_convention.md)
2. MD Gateway 未啟動 → 執行 `docker exec godzilla-dev pm2 list`
3. 訂閱參數錯誤 → 檢查 `InstrumentType` 和 `Exchange` 是否正確

---

### 問題: IndexError - list index out of range (下單時)

**根本原因**: Depth 數據尚未初始化,`bid_price[0]` 為空

**解決方案**: [depth_object_contract.md - 使用陷阱](../contracts/depth_object_contract.md#使用陷阱)
```python
def on_depth(self, context, depth):
    # ❌ 錯誤: 直接使用可能為空
    price = depth.bid_price[0]

    # ✅ 正確: 先檢查是否有效
    if depth.bid_price[0] > 0:
        price = depth.bid_price[0]
```

---

### 問題: TD Gateway 登入失敗 (Invalid API-key)

**排查步驟**: [TESTNET.md - API Keys 驗證](TESTNET.md#驗證-api-keys)

**檢查清單**:
1. API Key 是否從測試網申請? (非主網)
2. 配置檔路徑正確? (`~/.config/kungfu/app/runtime/config/td/binance/<account>.json`)
3. `access_key` 和 `secret_key` 是否完整複製? (無多餘空格)
4. 測試網 URL 是否正確? (`https://testnet.binance.vision`)

---

### 問題: PM2 服務一直重啟 (restart loop)

**排查步驟**: [pm2_startup_guide.md - 故障排除](../operations/pm2_startup_guide.md#故障排除)

**常見原因**:
1. 啟動順序錯誤 → Master 必須先啟動
2. 配置檔損壞 → 檢查 JSON 格式是否正確
3. 埠號衝突 → 檢查是否有多個實例運行

---

## 📚 進階學習資源

### 深入理解案例 (可選)

如需深入理解系統架構和常見陷阱,可閱讀:
- [operations/debugging_case_studies.md](../operations/debugging_case_studies.md) - 真實除錯案例分析 (進階學習資源)
  - 案例1: PM2 + 數據庫配置問題 (實用價值高)
  - 案例2: 兩個數據庫路徑衝突 (常見陷阱)

### Python/C++ 綁定細節

如需理解 Python 如何調用 C++ 核心:
- [modules/python_bindings.md](../modules/python_bindings.md) - pybind11 綁定規則
- [CODE_INDEX.md](../CODE_INDEX.md#Python-綁定) - 綁定程式碼位置

### 完整安裝指南

如需從零開始安裝環境:
- [archive/INSTALL.md](INSTALL.md) - 完整安裝步驟
- [archive/HACKING.md](HACKING.md) - 開發環境設定

---

## 🔄 文檔導航

**需要更詳細的導航?** 查看:
- [NAVIGATION.md](../NAVIGATION.md) - 完整的任務導向索引
- [REFERENCE.md](../REFERENCE.md) - 文檔系統概覽
- [CODE_INDEX.md](../CODE_INDEX.md) - 程式碼錨點索引

**需要快速指令?** 查看:
- [QUICK_START.md](../operations/QUICK_START.md) - 所有常用指令集錦

---

## 📝 學習路徑更新

**更新時間**: 2025-12-01
**適用版本**: godzilla-evan v2.0+
**預估 Token**: ~2500

如有問題或建議,請參考 [NAVIGATION.md](../NAVIGATION.md) 找到對應文檔或提交 Issue。
