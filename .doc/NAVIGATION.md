# .doc 導航系統

## 一、我想做什麼? (任務導向索引)

### 🎯 開發新策略
**閱讀順序**:
1. `modules/strategy_framework.md` - 生命週期、回調函數、Context API
2. `contracts/strategy_context_api.md` - 完整 API 參考手冊
3. `operations/debugging_guide.md` - 除錯技巧與常見問題

**Token 預算**: ~15k
**適合對象**: Python 開發者,具基本交易知識
**範例程式**: `strategies/demo_spot.py`

---

### 🔍 理解事件流與架構
**閱讀順序**:
1. `modules/yijinjing.md` - 事件溯源基礎 (Journal 機制)
2. `modules/wingchun.md` - 交易引擎架構
3. `modules/event_flow.md` - 完整事件流程圖
4. `modules/order_lifecycle_flow.md` - 訂單狀態機

**Token 預算**: ~25k
**前置知識**: 需理解 event sourcing 概念
**依賴圖**: yijinjing → wingchun → (strategy + binance + ledger)

---

### 🐛 除錯 Binance 問題
**閱讀順序**:
1. `modules/binance_extension.md` - REST/WebSocket 實作細節
2. `config/CONFIG_REFERENCE.md` - 配置檔檢查清單與安全指南
3. `config/NAMING_CONVENTIONS.md` - 帳號與交易對命名規範
4. `archive/TESTNET.md` - 測試網設定與驗證

**Token 預算**: ~22k
**常見原因**: API key 錯誤、網路問題、市場類型配置錯誤
**快速驗證**: 檢查 `~/.config/kungfu/app/runtime/config/td/binance/`

---

### 🚀 部署與服務管理
**閱讀順序**:
1. `operations/QUICK_START.md` - 快速啟動指令
2. `operations/pm2_startup_guide.md` - PM2 完整操作指南
3. `operations/cli_operations_guide.md` - CLI 工具詳解
4. `operations/debugging_guide.md` - 服務診斷流程

**Token 預算**: ~18k
**前置條件**: Docker 容器已啟動
**啟動順序**: Master → Ledger → MD → TD → Strategy (每步間隔 5 秒)

---

### 🏗️ 新增交易所 Gateway
**閱讀順序**:
1. `modules/wingchun.md` - Gateway 介面定義
2. `modules/binance_extension.md` - 參考實作 (MarketData + Trader)
3. `contracts/binance_config_contract.md` - 配置範本
4. `adr/004-binance-market-toggle.md` - 架構決策參考

**Token 預算**: ~20k
**前置知識**: C++ 開發經驗,熟悉交易所 API
**實作檢查清單**: EXTENSION_REGISTRY_MD, EXTENSION_REGISTRY_TD

---

### 📊 修改核心資料結構
**閱讀順序**:
1. `CODE_INDEX.md` - 定位程式碼位置
2. `contracts/order_object_contract.md` - Order 結構與不變量
3. `contracts/depth_object_contract.md` - Depth 結構與不變量
4. `modules/python_bindings.md` - pybind11 綁定規則

**Token 預算**: ~12k
**影響範圍**: C++ 核心 + Python 綁定 + 策略層
**測試要求**: 必須更新單元測試與整合測試

---

## 二、關鍵字快速查找

| 關鍵字 | 主要文檔 | 補充文檔 |
|--------|---------|---------|
| **Order** | contracts/order_object_contract.md | modules/order_lifecycle_flow.md, CODE_INDEX.md |
| **Depth** | contracts/depth_object_contract.md | modules/binance_extension.md#websocket |
| **Journal** | modules/yijinjing.md | modules/event_flow.md |
| **Context API** | contracts/strategy_context_api.md | modules/strategy_framework.md#callbacks |
| **PM2** | operations/pm2_startup_guide.md | operations/QUICK_START.md |
| **配置** | config/CONFIG_REFERENCE.md | config/NAMING_CONVENTIONS.md |
| **Binance** | modules/binance_extension.md | contracts/binance_config_contract.md, adr/004-binance-market-toggle.md |
| **除錯案例** | operations/debugging_case_studies.md | operations/debugging_guide.md |
| **策略生命週期** | modules/strategy_framework.md | modules/strategy_lifecycle_flow.md |
| **持倉管理** | modules/ledger_system.md | contracts/order_object_contract.md#position |
| **Python 綁定** | modules/python_bindings.md | CODE_INDEX.md#pybind |

---

## 三、文檔依賴關係圖

```
基礎層 (必讀)
  └─ yijinjing.md ────────────── 事件溯源機制
           │
核心層 (架構)
  └─ wingchun.md ─────────────── 交易引擎
           │
      ┌────┴────┬─────────┬─────────┬──────────┐
      │         │         │         │          │
   strategy  binance   ledger   python    event_flow
   framework extension  system  bindings
      │         │                  │
      │         │                  │
   context   config            pybind
     api    contract          綁定規則
      │         │
      │         │
   callbacks  dangerous
   時序圖      keys
```

**建議學習路徑**:
1. **新手**: yijinjing → wingchun → strategy_framework → context_api
2. **除錯**: 直接查對應模組 (binance/ledger/etc.) + debugging_guide
3. **架構研究**: yijinjing → wingchun → event_flow → order_lifecycle_flow

---

## 四、文檔狀態追蹤

| 文檔 | 狀態 | 最後驗證 | 對應程式碼版本 | 備註 |
|------|------|---------|---------------|------|
| **contracts/order_object_contract.md** | ✅ 已驗證 | 2025-11-17 | msg.h:666-730 | 狀態機完整 |
| **modules/binance_extension.md** | ✅ 已驗證 | 2025-11-20 | ADR-004 實作後 | 市場切換功能已更新 |
| **operations/pm2_startup_guide.md** | ✅ 已驗證 | 2025-11-18 | - | 操作流程正確 |
| **operations/debugging_case_studies.md** | ✅ 已驗證 | 2025-12-01 | - | 學習資源 (非操作指南) |
| **contracts/depth_object_contract.md** | ✅ 已驗證 | 2025-11-17 | msg.h:242-302 | 陷阱說明清楚 |
| **modules/strategy_framework.md** | ⚠️ 待驗證 | 2025-10-15 | strategy.py:35-184 | 可能有新 API |
| **modules/yijinjing.md** | ✅ 已驗證 | 2025-11-10 | - | 核心機制穩定 |
| **config/CONFIG_REFERENCE.md** | ✅ 已驗證 | 2025-12-01 | - | 統一配置參考 |
| **config/NAMING_CONVENTIONS.md** | ✅ 已驗證 | 2025-12-01 | - | 命名規範統一 |
| **archive/TESTNET.md** | ✅ 已驗證 | 2025-11-20 | - | 測試網流程完整 |

**圖例**:
- ✅ 已驗證: 文檔與程式碼同步,可放心使用
- ⚠️ 待驗證: 可能有更新,使用時注意核對程式碼
- ❌ 過時: 需要重寫,暫時不要使用

---

## 五、Token 預算管理

### 按任務類型估算

| 任務類型 | 推薦文檔數 | 預估 Tokens | 適合場景 |
|---------|-----------|-------------|---------|
| **快速查詢** | 1-2 個 | 5-8k | 查 API 用法、確認配置 |
| **開發任務** | 2-3 個 | 12-18k | 寫新策略、修改功能 |
| **深度除錯** | 3-5 個 | 20-30k | 複雜問題診斷 |
| **架構研究** | 5-8 個 | 35-50k | 理解整體設計 |
| **全量載入** | 全部 36 個 | ~576k | 僅在必要時 (不推薦) |

### 載入策略建議

1. **冷啟動** (首次接觸專案):
   - 讀 `CLAUDE.md` + `NAVIGATION.md` = ~800 tokens
   - 建立系統級心智模型

2. **一般開發**:
   - 根據任務查 NAVIGATION.md 的「我想做什麼」
   - 按推薦順序載入 2-3 個文檔
   - 避免一次載入超過 30k tokens

3. **複雜任務**:
   - 先載入依賴圖的基礎層 (yijinjing + wingchun)
   - 再載入任務相關文檔
   - 分批載入,避免 context window 浪費

4. **禁止行為**:
   - ❌ 未讀 NAVIGATION.md 就直接猜測檔案路徑
   - ❌ 一次載入全部文檔 (除非真的需要全局理解)
   - ❌ 引用未實際讀取的文檔內容

---

## 六、文檔維護指南

### 修改程式碼後的更新流程

1. **資料結構變更** (msg.h):
   - 更新 `contracts/*_object_contract.md`
   - 更新 `CODE_INDEX.md` 的行號
   - 檢查 `modules/python_bindings.md` 是否需同步

2. **策略 API 變更** (context.cpp, strategy.py):
   - 更新 `contracts/strategy_context_api.md`
   - 更新 `modules/strategy_framework.md`
   - 更新範例程式 `strategies/demo_*.py`

3. **配置格式變更**:
   - 更新 `config/CONFIG_REFERENCE.md`
   - 更新 `config/NAMING_CONVENTIONS.md` (若影響命名)
   - 更新 `contracts/binance_config_contract.md` (若影響 Binance)
   - 更新 `config/examples/` 的範例檔

4. **重大架構決策**:
   - 創建新 `adr/00X-decision-name.md`
   - 更新受影響的 modules 文檔
   - 在 NAVIGATION.md 添加交叉引用

### 文檔驗證清單

- [ ] 所有程式碼引用 (file:line) 是否正確
- [ ] 依賴關係圖是否更新
- [ ] NAVIGATION.md 的 token 估算是否準確
- [ ] 文檔狀態追蹤表是否更新
- [ ] 交叉引用連結是否正常
