# on_factor 和 on_order 回調失敗修復分析

## 問題總結

### 問題 1: on_factor 回調不執行（確定根因）

**症狀**:
- C++ 調用 12+ 次 `on_factor_callback()`，日誌顯示 "Calling strategy on_factor"
- Python `on_factor()` 函數內部代碼完全不執行（連第一行 debug log 都沒有）

**根因**（85% 確定）:
- 文件: `core/cpp/wingchun/pybind/pybind_wingchun.cpp:216-220`
- on_factor 是**唯一**使用 `py::gil_scoped_acquire` 的回調
- 主線程已持有 GIL，雙重獲取導致 PYBIND11_OVERRIDE 靜默失敗

**證據**:
```cpp
// pybind_wingchun.cpp:216-220 (問題代碼)
void on_factor(...) override {
    py::gil_scoped_acquire acquire;  // ⚠️ 問題所在
    PYBIND11_OVERRIDE(void, strategy::Strategy, on_factor, ...);
}

// pybind_wingchun.cpp:189 (正常工作的對照)
void on_depth(...) override {
    PYBIND11_OVERLOAD(void, strategy::Strategy, on_depth, ...);
}
```

**其他回調都沒有 GIL acquire，且工作正常**:
- on_depth: 500+ 次調用成功
- on_order: （應該能工作，但有其他問題）
- on_transaction: 正常
- on_position: 正常

---

### 問題 2: on_order 回調不傳播（推測根因）

**症狀**:
- TD Gateway 收到 Binance 訂單更新（ex_order_id=10957870579）
- Python 策略零 on_order 回調
- 事件未到達 Runner 的 on_order 處理邏輯

**推測根因**:
- 文件: `core/cpp/wingchun/src/strategy/runner.cpp:338-355`
- 雙重過濾機制可能失效：
  1. `to(context_->app_.get_home_uid())` - Order 事件的 dest 必須匹配
  2. `order.strategy_id == strategy.first` - strategy_id 必須匹配

**關鍵代碼**:
```cpp
// runner.cpp:338-355
events_ | is(msg::type::Order) | to(context_->app_.get_home_uid()) |
$([&](event_ptr event) {
    auto order = event->data<Order>();
    for (const auto &strategy : strategies_) {
        if (order.strategy_id == strategy.first) {
            strategy.second->on_order(context_, order);
            break;
        }
    }
});
```

**可能的失敗點**:
1. **dest 不匹配**: Order 事件的 dest 沒有設置為 Strategy 的 home_uid
2. **strategy_id 不匹配**: Order 的 strategy_id 沒有正確設置
3. **寫入問題**: TD Gateway 沒有正確寫入 Order 事件

**TD Gateway 設置 dest/strategy_id 的邏輯**:
- 文件: `core/extensions/binance/src/trader_binance.cpp:409-434`
- 使用 `get_writer(source)` 獲取 writer
- `source` 來自 insert_order 時的 event->source()

---

## 關鍵架構知識

### GIL (Global Interpreter Lock) 管理

**pybind11 GIL 規則**:
1. **從 C++ 調用 Python 函數**（需要 GIL）:
   - 使用 `py::gil_scoped_acquire` 獲取 GIL
   - 例如: 從 C++ 線程調用 Python 回調

2. **從 Python 調用 C++ 函數**（已持有 GIL）:
   - 不需要再獲取 GIL（會導致死鎖或失敗）
   - 例如: Python override C++ 虛函數

3. **本項目的情況**:
   - Runner::on_factor_callback() 在**主線程**中運行
   - 主線程已持有 GIL（因為 Python 策略在主線程運行）
   - PyStrategy::on_factor 是 PYBIND11_OVERRIDE，不應獲取 GIL

### 事件路由機制

**Order 事件路由流程**:
```
TD Gateway (trader_binance.cpp)
    ↓ writer->open_data<Order>()
    ↓ writer->close_data()  // 發送到 Journal
    ↓
Yijinjing Journal
    ↓
Strategy Runner (runner.cpp)
    ↓ events_ | is(msg::type::Order) | to(home_uid)
    ↓ filter by strategy_id
    ↓
PyStrategy::on_order
```

**關鍵驗證點**:
1. TD Gateway 是否正確設置 Order.strategy_id
2. TD Gateway 是否正確設置事件的 dest
3. Runner 的 home_uid 是否正確

---

## 修復優先級

### Priority 1: on_factor（阻塞，確定性高）

**根因確定度**: 85%
**修復複雜度**: 低
**影響範圍**: 僅 pybind_wingchun.cpp 一處

### Priority 2: on_order（阻塞，需診斷）

**根因確定度**: 60%
**修復複雜度**: 中（需先診斷）
**影響範圍**: 可能涉及 runner.cpp + trader_binance.cpp

---

## 下一步

設計分階段修復計劃，包括：
1. on_factor 的直接修復（移除 GIL acquire）
2. on_order 的診斷方案（添加 debug 日誌）
3. on_order 的修復方案（基於診斷結果）
4. 測試和驗證方法
5. Git 提交策略
