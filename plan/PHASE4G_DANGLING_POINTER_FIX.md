# Phase 4G: Dangling Pointer Fix - Implementation & Testing Guide

**Status**: ✅ Fix implemented (commit f2a0be2), ⏳ Testing required
**Date**: 2025-12-12
**Priority**: P0 (生產級穩定性必須)

---

## 問題回顧

**File**: `hf-live/adapter/signal_api.cpp:57-66`
**Root Cause**: 局部 `std::vector<double> predictions` 在 lambda 結束時銷毀 → `predictions.data()` 變成懸空指針
**Symptom**: `double free or corruption (!prev)` 在 Python `on_factor` 回調成功執行**之後**

---

## 修復方案 (Option A - 已實施)

**File**: `hf-live/_comm/signal_sender.h:59`

**核心修改**:
```cpp
// ✅ 修復：立即複製數據到本地 vector
std::vector<double> values_copy(values, values + count);

callback_(symbol, timestamp, values_copy.data(), count, user_data_);
// values_copy 在這裡析構,但 callback 已安全執行完畢
```

**Performance Impact**:
- Copy overhead: ~30ns (2 double values)
- CPU impact: < 0.01%
- Memory: No increase (local variable)

---

## 編譯與部署

### Step 1: 編譯 libsignal.so (在容器內)

```bash
# 進入容器
docker exec -it godzilla-dev bash

# 編譯
cd /app/hf-live/build
make clean
make -j4

# 驗證編譯產物
ls -lh libsignal.so
# Expected: -rwxr-xr-x 1 root root 9.4M ...

# 驗證符號
nm -C libsignal.so | grep "SignalSender::Send"
# Expected: Multiple Send() symbols
```

### Step 2: 重啟策略

```bash
# 方案 A: PM2 restart (快速測試)
pm2 restart strategy-test-hf-live

# 方案 B: 完整重啟 (推薦,確保乾淨狀態)
pm2 stop all
pm2 delete all
cd /app/scripts/test_hf_live && ./clean.sh

# 按順序啟動 (間隔 5 秒)
pm2 start /app/scripts/binance_test/master.json && sleep 5
pm2 start /app/scripts/binance_test/ledger.json && sleep 5
pm2 start /app/scripts/binance_test/md_binance.json && sleep 5
pm2 start /app/scripts/test_hf_live/strategy.json
```

---

## 測試計劃

### P0 Test: 60-Second Basic Functionality ⭐⭐⭐

**Goal**: 無 "double free or corruption" 錯誤

```bash
# 等待 60 秒
sleep 60

# 檢查記憶體錯誤
tail -200 /root/.pm2/logs/strategy-test-hf-live-error.log | grep -i "free\|corruption\|segmentation"

# 預期: 無匹配 (如果有輸出 = 測試失敗)
```

**Success Criteria**:
- ✅ 無 "double free" 錯誤
- ✅ 無 "corruption" 錯誤
- ✅ PM2 restart count = 0
- ✅ 看到 📊 和 🔢 emoji (數據流正常)

---

### P1 Test: 2-Hour Stress Test ⭐⭐

**Goal**: 零異常重啟,記憶體穩定

```bash
# 記錄初始 restart count
pm2 list | grep strategy-test-hf-live

# 等待 2 小時
sleep 7200

# 檢查 restart count
pm2 list | grep strategy-test-hf-live
# Expected: ↺ 0 (no increase)

# 檢查記憶體
pm2 list | grep strategy-test-hf-live
# Expected: mem ~140-170 MB (stable)
```

**Success Criteria**:
- ✅ Zero restarts in 2 hours
- ✅ Memory stable (~140-170 MB)
- ✅ No memory errors in logs
- ✅ Python on_factor 回調成功 (看到 🎊)

---

### P2 Test: 17+ Hour Stability Test (Optional) ⭐

**Goal**: 與 Phase 4C 相同的穩定性標準

```bash
# 運行 17 小時
# (overnight test)

# 檢查最終狀態
pm2 list
tail -500 /root/.pm2/logs/strategy-test-hf-live-error.log
```

**Success Criteria**:
- ✅ 17+ hours runtime
- ✅ Zero restarts
- ✅ Zero memory errors
- ✅ 符合生產級穩定性標準

---

## 快速檢查清單

**編譯階段**:
- [ ] `make clean && make -j4` 無錯誤
- [ ] `libsignal.so` 大小 ~9.4 MB
- [ ] `nm -C libsignal.so | grep Send` 有輸出

**部署階段**:
- [ ] 完整系統重啟 (clean.sh + 按順序啟動)
- [ ] 等待至少 10 秒讓服務穩定

**P0 測試** (必須通過):
- [ ] 運行 60 秒無 "double free" 錯誤
- [ ] PM2 restart count = 0
- [ ] 看到 📊 emoji (數據流正常)

**P1 測試** (強烈建議):
- [ ] 運行 2 小時無重啟
- [ ] 記憶體穩定 ~140-170 MB
- [ ] 看到 🎊 emoji (Python 回調成功)

---

## 預期日誌序列 (成功案例)

```
🏁 [test0000::FactorEntry] Created for: BTCUSDT
📊 [test0000 #10] bid=90279.0 ask=90279.9
📊 [test0000 #100] bid=90306.9 ask=90310.7
🔢 [test0000::UpdateFactors] spread=3.8 mid=90308.8
📤 [FactorThread] Pushed result to queue
🚀 [ScanThread::SendData] Sending factors for BTCUSDT
📥 [ModelEngine] Received factors for BTCUSDT
🤖 [test0000::Model] Created with 3 factors
🔮 [test0000::Calculate] asset=BTCUSDT → output=[1, 0.8]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📨 [SignalSender::Send] CALLED!
   Symbol: BTCUSDT
   Count: 2
   Callback: VALID
   Values: [1, 0.8]
   ✅ Calling callback (with safe data copy)...  ← 關鍵修改!
   ✅ Callback returned
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[FACTOR] 🎊 Received factor for BTCUSDT @ 1765377407481907263 (count=2)
[FACTOR] Calling strategy on_factor for strategy_id=1350253488
```

**關鍵差異**: 現在日誌顯示 "with safe data copy",表示修復已生效

---

## 失敗處理

| 症狀 | 可能原因 | 解決方案 |
|------|---------|---------|
| 仍然 "double free" | 編譯未生效 | 確認 libsignal.so 時間戳,重新編譯 |
| PM2 異常重啟 | 其他記憶體問題 | 檢查 Phase 4C 的 3 個修復是否完整 |
| 無 emoji 日誌 | 數據流未觸發 | 檢查 MarketEventProcessor 設定 |
| Callback NULL | 綁定問題 | 檢查 signal_register_callback 日誌 |

---

## Git Commits

**子模組** (hf-live):
```
f2a0be2 - fix(signal_sender): resolve dangling pointer issue in Send()
```

**主倉庫**:
```
3e4beb6 - chore: update hf-live submodule (dangling pointer fix)
```

---

## 相關文檔

- **問題定位**: `plan/prd_hf-live.10-e2e-testing.md` (Phase 4F Issue #1)
- **修復計劃**: `plan/prd_hf-live.11-implementation-history.md`
- **Phase 4C 修復**: `plan/debug_hf-live.00-memory-corruption-fix.md`

---

**預計測試時間**: P0 (2 分鐘) + P1 (2 小時) = ~2 小時
**風險級別**: 極低 (局部修改,清晰修復方案)
**優先級**: P0 (生產級穩定性必須)
