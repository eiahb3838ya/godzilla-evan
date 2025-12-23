# Debug Plan: test_hf_live 帳號註冊機制調查

## 調查目標
深入理解為什麼 Phase 4B 的 test_hf_live 沒有明確調用 `context.add_account()`，但訂單仍能正常工作。

---

## 核心疑問

### 問題 1: Phase 4B 如何在沒有 add_account() 的情況下工作？
```python
# Phase 4B (commit 5e2f708) - 能工作的版本
def pre_start(context):
    context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)
    # ❓ 沒有 add_account()，為什麼能下單？

def on_depth(context):
    order_id = context.insert_order(
        source="binance",
        account="gz_user1",  # ❓ 這個帳號是怎麼被註冊的？
        ...
    )
```

### 問題 2: 可能的隱式註冊機制
需要調查以下可能性：
1. **配置文件自動註冊**: runner.cpp 或 context.cpp 是否在啟動時自動註冊了 config.json 中的帳號？
2. **TD Gateway 連接觸發**: 當 TD Gateway 啟動時是否自動註冊帳號？
3. **insert_order 隱式註冊**: insert_order() 內部是否有自動註冊邏輯？
4. **Spot vs Futures 行為差異**: Spot 和 Futures 的帳號驗證邏輯是否不同？

### 問題 3: Phase 6 為什麼會失敗？
需要找出 Phase 4B → Phase 6 之間的變化：
1. Wingchun 核心代碼變化
2. 配置處理邏輯變化
3. 帳號驗證邏輯加強
4. InstrumentType.Spot → InstrumentType.FFuture 的影響

---

## 調查計劃

### 階段 1: Git 歷史深度分析
**目標**: 找出所有相關的 commit 和代碼變化

#### 1.1 完整的 test_hf_live 演變史
```bash
# 查看完整歷史
git log --all --oneline --graph -- strategies/test_hf_live/

# 找出每個階段的 commit
git log --grep="phase-4" --all --oneline
git log --grep="phase-5" --all --oneline
git log --grep="phase-6" --all --oneline

# 檢查每個階段的完整代碼
git show 5e2f708:strategies/test_hf_live/test_hf_live.py  # Phase 4B
git show a2d8f5f:strategies/test_hf_live/test_hf_live.py  # Phase 4B fix
git show 419c489:strategies/test_hf_live/test_hf_live.py  # Phase 6
```

#### 1.2 runner.cpp 的變化
```bash
# 查看 runner.cpp 的歷史
git log --oneline --all -- core/cpp/wingchun/src/strategy/runner.cpp | head -20

# 檢查配置處理邏輯的變化
git diff 5e2f708..419c489 -- core/cpp/wingchun/src/strategy/runner.cpp

# 尋找帳號相關的邏輯
git log -S "add_account" --all -- core/cpp/wingchun/
git log -S "account_location_ids" --all -- core/cpp/wingchun/
```

#### 1.3 context.cpp 的變化
```bash
# 查看 context.cpp 的歷史
git log --oneline --all -- core/cpp/wingchun/src/strategy/context.cpp | head -20

# 檢查 add_account 和驗證邏輯的變化
git diff 5e2f708..419c489 -- core/cpp/wingchun/src/strategy/context.cpp

# 查看 insert_order 的實現變化
git log -S "insert_order" --all -- core/cpp/wingchun/src/strategy/context.cpp
```

---

### 階段 2: 源碼深度分析
**目標**: 理解帳號註冊和驗證的完整流程

#### 2.1 runner.cpp 的配置處理
**文件**: `core/cpp/wingchun/src/strategy/runner.cpp`

**關鍵檢查點**:
1. `Runner::run()` 函數中的配置加載邏輯
2. 是否有自動調用 `context.add_account()` 的代碼
3. 如何處理 config.json 中的 `account` 和 `td_source` 字段

**調查問題**:
```cpp
// 在 runner.cpp 中查找：
// 1. 是否有類似這樣的代碼？
if (config.has("account") && config.has("td_source")) {
    context->add_account(config["td_source"], config["account"]);
}

// 2. 或者是否有自動註冊邏輯？
auto accounts = config.get_accounts();
for (auto& account : accounts) {
    context->add_account(account.source, account.name);
}
```

#### 2.2 context.cpp 的帳號管理
**文件**: `core/cpp/wingchun/src/strategy/context.cpp`

**關鍵檢查點**:
1. `add_account()` 函數的完整實現
2. `account_location_ids_` 映射的初始化
3. `lookup_account_location_id()` 的調用時機
4. `insert_order()` 中的帳號驗證邏輯

**調查問題**:
```cpp
// 在 context.cpp 中查找：
// 1. insert_order 是否有隱式註冊？
uint64_t Context::insert_order(..., const std::string& account, ...) {
    // ❓ 是否有這樣的代碼？
    if (account_location_ids_.find(account_id) == account_location_ids_.end()) {
        // 自動註冊？
        add_account(...);
    }
}

// 2. 是否有其他隱式註冊入口？
// - 在構造函數中？
// - 在 init() 中？
// - 在 pre_start 之前的某個階段？
```

#### 2.3 Spot vs Futures 的差異
**文件**:
- `core/cpp/wingchun/src/strategy/context.cpp`
- `core/extensions/binance/src/trader_binance.cpp`

**關鍵檢查點**:
1. InstrumentType.Spot 和 InstrumentType.FFuture 的處理邏輯差異
2. Spot 訂單是否有更寬鬆的帳號驗證
3. Futures 訂單是否有額外的帳號檢查

**調查問題**:
```cpp
// 查找 InstrumentType 相關的帳號驗證邏輯
if (instrument_type == InstrumentType::FFuture) {
    // ❓ Futures 是否有更嚴格的驗證？
    validate_account_registration();
}
```

---

### 階段 3: 實驗驗證
**目標**: 通過實驗確認理論假設

#### 3.1 測試 Phase 4B 原始代碼
```bash
# 切換到 Phase 4B commit
git checkout 5e2f708

# 重新編譯
cd core/cpp && rm -rf build && mkdir build && cd build
cmake .. && make -j$(nproc)

# 啟動測試
cd /app/scripts/binance_test
./run.sh start
sleep 25
pm2 start strategy_test_hf_live.json

# 觀察日誌
pm2 logs strategy_test_hf_live | grep -E "Order|account"
```

**檢查項**:
- [ ] 訂單是否成功提交？
- [ ] 是否有 "invalid account" 錯誤？
- [ ] 日誌中是否有自動註冊的信息？

#### 3.2 添加調試日誌
在關鍵位置添加日誌輸出：

**context.cpp 中**:
```cpp
void Context::add_account(...) {
    std::cerr << "[DEBUG] add_account called: " << account << "@" << source << std::endl;
    // ... 原有代碼
}

uint32_t Context::lookup_account_location_id(...) {
    std::cerr << "[DEBUG] lookup_account_location_id: " << account << std::endl;
    std::cerr << "[DEBUG] account_location_ids_ size: " << account_location_ids_.size() << std::endl;
    // ... 原有代碼
}
```

**runner.cpp 中**:
```cpp
void Runner::run() {
    std::cerr << "[DEBUG] Runner::run - config: " << config.dump() << std::endl;
    // ... 檢查是否有自動 add_account
}
```

#### 3.3 對比測試
1. **測試 A**: Phase 4B 原始代碼（Spot，無 add_account）
2. **測試 B**: Phase 6 原始代碼（Futures，無 add_account）
3. **測試 C**: Phase 6 + add_account（Futures，有 add_account）

記錄每個測試的結果，找出差異點。

---

### 階段 4: 尋找隱式註冊機制
**目標**: 找出 Phase 4B 能工作的真正原因

#### 4.1 可能的隱式註冊入口點
需要檢查以下文件和函數：

1. **runner.cpp**
   - `Runner::run()` - 策略主入口
   - `Runner::init()` - 初始化邏輯
   - 配置處理相關的函數

2. **context.cpp**
   - `Context::Context()` - 構造函數
   - `Context::pre_start()` - pre_start 觸發前的邏輯
   - 任何在 insert_order 之前會調用的函數

3. **strategy.cpp / strategy.py**
   - Python 層是否有自動註冊邏輯
   - pybind11 綁定中是否有特殊處理

4. **service.cpp / hero.cpp**
   - 服務啟動時是否有帳號自動發現
   - TD Gateway 連接時是否觸發註冊

#### 4.2 配置文件驅動的註冊
檢查是否有這樣的邏輯：
```cpp
// 偽代碼
void auto_register_accounts(Config& config) {
    if (config.has("account") && config.has("td_source")) {
        // 自動從配置註冊帳號
        context.add_account(config["td_source"], config["account"]);
    }
}
```

#### 4.3 TD Gateway 事件驅動的註冊
檢查是否有這樣的邏輯：
```cpp
// 當 TD Gateway 連接成功時
void on_td_connected(const std::string& source, const std::string& account) {
    // 自動註冊帳號
    context.add_account(source, account);
}
```

---

## 預期發現

### 假設 A: 存在隱式自動註冊機制
**如果成立**:
- Phase 4B 能工作是因為系統自動從 config.json 註冊了帳號
- Phase 6 失敗是因為這個機制被移除或修改了
- **修復方案**: 恢復自動註冊邏輯，或明確調用 add_account()

### 假設 B: Spot vs Futures 驗證邏輯不同
**如果成立**:
- Phase 4B (Spot) 的帳號驗證更寬鬆，允許未註冊的帳號
- Phase 6 (Futures) 的帳號驗證更嚴格，必須先註冊
- **修復方案**: 統一驗證邏輯，或根據 InstrumentType 調整策略

### 假設 C: 系統版本變化導致行為改變
**如果成立**:
- Wingchun 或 Yijinjing 在 Phase 4B → Phase 6 期間有重大更新
- 新版本加強了帳號驗證邏輯
- **修復方案**: 適應新版本要求，明確調用 add_account()

### 假設 D: 測試環境差異
**如果成立**:
- Phase 4B 測試時的環境與當前環境不同
- 可能是數據庫狀態、TD Gateway 配置等差異
- **修復方案**: 檢查環境配置，確保一致性

---

## 輸出成果

### 1. 調查報告
**文件**: `plan/plan/debug_hf-live.03-account-registration-report.md`

**內容包括**:
- Phase 4B 能工作的真正原因
- Phase 6 失敗的根本原因
- 完整的代碼變化對比
- 帳號註冊機制的完整流程圖

### 2. 修復建議
基於調查結果，提供以下修復建議：
1. **短期方案**: 立即可用的修復代碼
2. **長期方案**: 系統級的改進建議
3. **最佳實踐**: 策略開發的帳號管理指南

### 3. 測試用例
創建測試用例驗證修復：
- 不同 InstrumentType 的測試
- 有/無 add_account() 的對比測試
- 不同配置的測試

---

## Subagent 執行計劃

### Subagent 1: Git 歷史分析
**任務**: 執行階段 1 的所有 git 命令
**輸出**: 完整的代碼變化列表和差異

### Subagent 2: 源碼深度分析
**任務**: 執行階段 2 的源碼檢查
**輸出**: 關鍵函數的實現分析和可能的隱式註冊機制

### Subagent 3: 實驗驗證（可選）
**任務**: 如果前兩個 agent 無法確定答案，進行實驗驗證
**輸出**: 實驗結果和結論

---

## 成功標準

完成調查後，應能回答以下問題：
1. ✅ Phase 4B 為什麼沒有 add_account() 但能工作？
2. ✅ Phase 6 為什麼必須要 add_account()？
3. ✅ 是否存在隱式自動註冊機制？
4. ✅ Spot 和 Futures 的帳號驗證邏輯有何不同？
5. ✅ 如何修復才是最正確的方式？

---

**創建時間**: 2025-12-18
**狀態**: 等待批准
**預計執行時間**: 30-45 分鐘
---

## 調查結果報告 (2025-12-18 完成)

### 執行摘要

✅ **調查完成**: 通過 Git 歷史分析和源碼深度分析，確定了問題根因

✅ **假設驗證結果**:
- ❌ 假設 A (隱式自動註冊機制) - **被否決**
- ❌ 假設 B (Spot vs Futures 驗證差異) - **被否決**
- ❌ 假設 C (系統版本變化) - **部分正確但非根因**
- ✅ **真正原因**: Phase 6 重構時意外移除了 Phase 4B fix 中已添加的 `add_account()` 調用

### 關鍵發現

#### 發現 1: "能工作的 Phase 4B" 實際上是 Phase 4B fix

**Git 歷史證據**:
- `5e2f708` (Phase 4B 原始): 有下單功能，但**沒有** add_account() - **這個版本有 bug**
- `a2d8f5f` (Phase 4B fix): **添加了** add_account() - **這才是能工作的版本！**
- `419c489` (Phase 6): 重構時**移除了** add_account() - **退化成 bug 狀態**

**Phase 4B fix (a2d8f5f) 的代碼**:
```python
def pre_start(context):
    config = context.get_config()
    context.add_account(config["td_source"], config["account"])  # ✅ 關鍵行
    context.subscribe(...)
```

**用戶的記憶是正確的**: 他記得的是 Phase 4B fix 能工作，而非 Phase 4B 原始版本。

#### 發現 2: 不存在任何隱式註冊機制

**源碼分析結果** (`core/cpp/wingchun/src/strategy/`):

1. **runner.cpp**:
   - ❌ 不會從 config.json 自動註冊帳號
   - ❌ 沒有任何自動調用 `context->add_account()` 的邏輯

2. **context.cpp**:
   - ❌ 構造函數不會自動註冊帳號
   - ❌ `insert_order()` 不會隱式註冊，而是直接拋出異常

**帳號驗證流程** (context.cpp):
```cpp
// 必須先調用這個建立映射
void Context::add_account(...) {
    account_location_ids_[account_id] = account_location->uid;
}

// insert_order 會檢查映射是否存在
uint32_t Context::lookup_account_location_id(...) {
    if (account_location_ids_.find(account_id) == account_location_ids_.end()) {
        throw wingchun_error("invalid account " + account);  // ← Phase 6 錯誤來源
    }
    return account_location_ids_[account_id];
}
```

**結論**: 帳號必須通過明確的 `context.add_account()` 調用來註冊，沒有任何自動機制。

#### 發現 3: Spot 和 Futures 使用相同的帳號驗證邏輯

**源碼證據**:
- `InstrumentType` 只影響交易所 API 路由選擇 (spot API vs futures API)
- 帳號驗證邏輯在 `context.cpp` 中統一處理，無 InstrumentType 分支
- Spot 和 Futures 都會調用 `lookup_account_location_id()`，使用相同的驗證邏輯

**結論**: Phase 4B 使用 Spot 能工作不是因為驗證寬鬆，而是因為它有 `add_account()` 調用。

### 成功標準驗證

- ✅ **問題 1**: Phase 4B 為什麼沒有 add_account() 但能工作？
  - **答**: 能工作的是 Phase 4B **fix** (a2d8f5f)，它**有** add_account()

- ✅ **問題 2**: Phase 6 為什麼必須要 add_account()？
  - **答**: 所有版本都必須，Phase 6 只是不小心移除了它

- ✅ **問題 3**: 是否存在隱式自動註冊機制？
  - **答**: 不存在，必須明確調用 `context.add_account()`

- ✅ **問題 4**: Spot 和 Futures 的帳號驗證邏輯有何不同？
  - **答**: 完全相同，無差異

- ✅ **問題 5**: 如何修復才是最正確的方式？
  - **答**: 在 pre_start() 中添加 `context.add_account(config["td_source"], config["account"])`，恢復到 Phase 4B fix 的狀態

### 修復方案

詳見 `/home/huyifan/.claude/plans/compiled-prancing-flame.md`

兩個修復：
1. **Fix-1**: 在 test_hf_live.py 添加 `context.add_account()` 調用
2. **Fix-2**: 在 model_calculation_engine.cc 將模型從 "test0000" 改為 "linear"

### 經驗教訓

1. **Git 歷史很重要**: 代碼演變過程中會有 bug 修復，不能只看最初版本
2. **不要假設隱式行為**: Kungfu 框架要求明確的帳號註冊，沒有自動化
3. **重構時要小心**: Phase 6 重構時意外移除了關鍵的 add_account() 調用
4. **測試很關鍵**: 如果 Phase 4B fix 有測試覆蓋，Phase 6 就不會退化

---

**調查時間**: 2025-12-18
**調查人員**: Claude Code (Sonnet 4.5)
**狀態**: ✅ 完成
**下一步**: 執行修復計畫
