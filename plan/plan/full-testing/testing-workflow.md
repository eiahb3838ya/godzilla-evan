# test_hf_live Phase 6 完整測試流程

## 任務目標
測試 Phase 6 實現（全市場數據 + 15 因子 + 線性模型），驗證完整數據流，觀察日誌輸出。

---

## 階段 0: Git 分支管理（前置步驟）

### 0.1 Stash 當前修改
```bash
git stash push -m "Phase 6: Full market data + 15 factors + linear model"
```

### 0.2 創建新分支
```bash
git checkout -b phase-6-full-market-data
```

### 0.3 應用 stash 並提交
```bash
git stash pop
git add .
git commit -m "feat(phase-6): implement full market data pipeline with linear model

- Extended runner.cpp to forward Ticker (102) and IndexPrice (104) to hf-live
- Extended signal_api.cpp to handle Ticker and IndexPrice events
- Created market factor module with 15 factors:
  * Depth factors (5): spread, mid_price, bid_ask_ratio, depth_imbalance, weighted_mid
  * Trade factors (5): trade_volume_ma, trade_direction, trade_intensity, vwap, trade_volatility
  * Ticker factors (3): ticker_spread, ticker_volume_ratio, ticker_momentum
  * IndexPrice factors (2): basis, basis_pct
- Implemented linear model for weighted factor combination
- Updated test_hf_live strategy to receive model outputs via on_factor callback
- Fixed factor registration includes (factor_entry_registry.h)
- Fixed hf::Side enum usage in Trade processing
- Fixed Ticker structure usage (single values, not arrays)

Data flow: Binance → MD → Journal → runner.cpp → libsignal.so → FactorEngine → LinearModel → on_factor (Python)
"
```

### 0.4 保存計劃文件
```bash
mkdir -p plan/plan/full-testing
cp /home/huyifan/.claude/plans/magical-sparking-treehouse.md plan/plan/full-testing/testing-workflow.md
git add plan/plan/full-testing/testing-workflow.md
git commit -m "docs(phase-6): add complete testing workflow plan"
```

---

## 數據流架構
```
Binance WebSocket → Wingchun MD → Journal → Strategy (runner.cpp)
                                                ↓
                        libsignal.so (hf-live)
                                                ↓
    Depth (101) → 5 factors ─┐
    Trade (103) → 5 factors ─┤
    Ticker (102) → 3 factors ├─→ FactorEngine → LinearModel
    IndexPrice (104) → 2 factors ─┘              ↓
                                        [pred_signal, pred_confidence]
                                                ↓
                                        on_factor (Python)
```

---

## 階段 1: 預檢查 (Pre-Flight Checks)

### 1.1 驗證 Docker 環境
```bash
# 進入容器
docker exec -it godzilla-dev bash

# 驗證工作目錄
cd /app
pwd  # 應該輸出: /app
```

### 1.2 驗證構建產物
```bash
# 檢查 libsignal.so 存在且是最新編譯
ls -lh /app/hf-live/build/libsignal.so
# 應該顯示: -rwxr-xr-x 1 root root 487K Dec 18 00:02 libsignal.so

# 驗證包含所有模組
nm /app/hf-live/build/libsignal.so | grep -E "(market|linear)" | head -5
# 應該能看到 market 和 linear 相關符號
```

### 1.3 驗證策略文件
```bash
# 檢查策略文件存在
ls -l /app/strategies/test_hf_live/test_hf_live.py
ls -l /app/strategies/test_hf_live/config.json

# 查看配置內容（確認 signal_library_path）
cat /app/strategies/test_hf_live/config.json
```

**預期輸出** (config.json):
```json
{
  "md_source": "binance",
  "td_source": "binance",
  "account": "gz_user1",
  "symbol": "btcusdt",
  "instrument_type": "FFuture",
  "signal_library_path": "/app/hf-live/build/libsignal.so"
}
```

### 1.4 驗證 Binance API 配置（重要！）
```bash
# 檢查 Binance 帳戶配置
ls ~/.config/kungfu/app/runtime/config/td/binance/gz_user1.json
```

**如果文件不存在**，需要創建：
```json
{
  "access_key": "YOUR_TESTNET_API_KEY",
  "secret_key": "YOUR_TESTNET_SECRET",
  "enable_spot": false,
  "enable_futures": true,
  "user_id": "gz_user1"
}
```

⚠️ **安全提醒**: 使用 Binance **Testnet** API Key，而非主網！

---

## 階段 2: 啟動服務 (Service Startup)

### 2.1 清理舊進程（如果需要）
```bash
# 查看當前進程
pm2 list

# 如果有舊進程，停止所有
cd /app/scripts/binance_test
./run.sh stop

# 等待 5 秒
sleep 5

# 驗證已停止
pm2 list  # 應該顯示空列表或所有進程已停止
```

### 2.2 使用標準啟動腳本
```bash
# 切換到腳本目錄
cd /app/scripts/binance_test

# 啟動基礎服務（Master → Ledger → MD → TD）
./run.sh start
```

**預期輸出**:
```
clearing journal...
starting master...
starting ledger...
starting md binance...
starting td...
```

### 2.3 等待服務註冊完成
```bash
# 等待 25 秒（5個服務 × 5秒間隔）
sleep 25

# 驗證所有服務運行中
pm2 list
```

**預期輸出** (pm2 list):
```
┌─────┬──────────────┬─────────┬─────────┬─────────┐
│ id  │ name         │ status  │ restart │ uptime  │
├─────┼──────────────┼─────────┼─────────┼─────────┤
│ 0   │ master       │ online  │ 0       │ 25s     │
│ 1   │ ledger       │ online  │ 0       │ 20s     │
│ 2   │ md_binance   │ online  │ 0       │ 15s     │
│ 3   │ td_binance:… │ online  │ 0       │ 10s     │
└─────┴──────────────┴─────────┴─────────┴─────────┘
```

### 2.4 啟動策略
```bash
# 啟動 test_hf_live 策略
pm2 start strategy_test_hf_live.json

# 等待 5 秒
sleep 5

# 驗證策略啟動
pm2 list | grep strategy_test_hf_live
```

---

## 階段 3: 日誌觀察 (Log Monitoring)

### 3.1 實時日誌監控（推薦方式）
```bash
# 在容器內開啟實時日誌（所有進程）
pm2 logs

# 或者只監控策略日誌
pm2 logs strategy_test_hf_live
```

**按 Ctrl+C 退出日誌流**

### 3.2 關鍵觀察點與時間線

#### T+0s: 策略啟動
**查看 C++ 日誌**:
```bash
tail -f /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live.log
```

**尋找關鍵行**:
```
[  info  ] [hero.cpp:143#register_location] registered location strategy/default/test_hf_live/live
[  info  ] [hero.cpp:164#register_channel] registered channel ...
```

✅ **檢查點 1**: 服務成功註冊到 Master

---

#### T+1s: hf-live 加載
**繼續監控 C++ 日誌**:
```bash
tail -f /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live.log
```

**尋找關鍵行**:
```
[  info  ] [runner.cpp:216#run] Signal library loaded successfully: /app/hf-live/build/libsignal.so
[  info  ] [runner.cpp:203#run] Signal callback registered successfully
```

✅ **檢查點 2**: hf-live 因子引擎已加載

**如果出現錯誤**:
```
[  error  ] cannot open shared object file: libsignal.so
```
→ 確認 libsignal.so 編譯成功（回到階段 1.2）

---

#### T+2s: Python 初始化
**查看 Python 日誌**:
```bash
tail -f /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live_py.log
```

**尋找關鍵行**:
```
[  info  ] [strategy.py:147#pre_start] pre start
[  info  ] 🏁 [Phase 6] Pre-Start - Testing Full Market Data + Linear Model
[  info  ] 📡 Subscribed: btcusdt (Futures) - All Market Data
```

✅ **檢查點 3**: 策略 pre_start 完成，已訂閱市場數據

---

#### T+3-10s: 市場數據接收
**Python 日誌中觀察 on_depth 回調**:
```bash
pm2 logs strategy_test_hf_live --lines 50 | grep -E "on_depth|on_factor"
```

**預期輸出**:
```
📊 [on_depth] btcusdt bid=96524.30 ask=96524.40 spread=0.10
📊 [on_depth] btcusdt bid=96525.10 ask=96525.20 spread=0.10
```

✅ **檢查點 4**: 接收到 Binance Depth 數據

---

#### T+5-15s: 因子計算輸出（關鍵！）
**Python 日誌中觀察 on_factor 回調**:
```bash
pm2 logs strategy_test_hf_live --lines 100 | grep "on_factor\|LinearModel"
```

**預期輸出**:
```
🤖 [LinearModel] btcusdt @ 1734480123456789000
   📈 Signal: +0.0523 (BULLISH)
   🎯 Confidence: 51.31%

🤖 [LinearModel] btcusdt @ 1734480124456789000
   ➡️  Signal: +0.0012 (NEUTRAL)
   🎯 Confidence: 50.03%
```

✅ **檢查點 5**: LinearModel 正常輸出預測信號

**信號解讀**:
- `pred_signal > 0.1`: 📈 BULLISH (看漲)
- `pred_signal < -0.1`: 📉 BEARISH (看跌)
- `-0.1 ≤ signal ≤ 0.1`: ➡️ NEUTRAL (中性)
- `pred_confidence`: 0.5-1.0（基於信號強度的 sigmoid）

---

#### T+10-30s: 測試訂單生命週期
**Python 日誌中觀察訂單流程**:
```bash
pm2 logs strategy_test_hf_live --lines 100 | grep -E "Order|order_id|ex_order_id"
```

**預期輸出序列**:
```
💸 [Placing Order] Buy 0.002 BTC @ 94593.4 (notional=189.19 USDT)
✅ [Order Placed] order_id=1234567890

📬 [on_order] order_id=1234567890 status=Submitted ex_order_id='12345678'

================================================================================
🎉🎉🎉 訂單已成功提交到 Binance Futures Testnet! 🎉🎉🎉

   📋 本地 Order ID: 1234567890
   🌐 Binance Order ID: 12345678
   💱 交易對: BTCUSDT (Futures)
   📊 方向: BUY (做多)
   📦 數量: 0.002 BTC

   ⏰ 訂單將保持 30 秒，請立即前往 Binance 網站確認！
   🌐 https://testnet.binancefuture.com
   👉 在 Open Orders 中查找 Order ID: 12345678
================================================================================

[30秒後...]
⏰ 30 秒已到，開始取消訂單...
🗑️  [Cancelling Order] order_id=1234567890 ex_order_id='12345678'
🎉 [Test Complete] Order cancelled successfully!
```

✅ **檢查點 6**: 訂單完整生命週期（提交 → 確認 → 取消）

---

## 階段 4: 數據流驗證清單

### 4.1 Market Data 流驗證
**檢查 MD 進程日誌**:
```bash
tail -n 50 /app/runtime/md/binance/binance/log/live/binance.log | grep -E "subscribe|depth|ticker"
```

**預期內容**:
- WebSocket 連接成功
- 訂閱確認: `btcusdt@depth5@100ms`
- 訂閱確認: `btcusdt@aggTrade`
- 訂閱確認: `btcusdt@ticker`
- 訂閱確認: `btcusdt@markPrice`

### 4.2 Factor Engine 初始化驗證
**在 C++ 日誌中查找因子註冊**:
```bash
grep -E "Factor|Model" /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live.log | head -20
```

**預期內容**:
```
Registered factor set: market (15 factors)
Registered model: linear (2 outputs)
```

### 4.3 完整數據流驗證
**使用以下命令統計回調次數**:
```bash
# 統計 on_depth 次數
grep -c "on_depth" /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live_py.log

# 統計 on_factor 次數
grep -c "LinearModel" /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live_py.log
```

**預期結果**:
- on_depth: 每秒 10-20 次（Binance 100ms 更新）
- on_factor: 與 on_depth 一致（每次 Depth 觸發因子計算）

---

## 階段 5: 延遲元數據觀察（可選）

### 5.1 重新編譯帶延遲監控的 hf-live
```bash
cd /app/hf-live
rm -rf build
mkdir build && cd build

# 啟用延遲元數據
cmake -DHF_TIMING_METADATA=ON ..
make -j$(nproc)

# 驗證編譯成功
ls -lh libsignal.so
```

### 5.2 重啟策略
```bash
# 停止策略
pm2 stop strategy_test_hf_live
pm2 delete strategy_test_hf_live

# 重新啟動
cd /app/scripts/binance_test
pm2 start strategy_test_hf_live.json
```

### 5.3 觀察延遲數據
```bash
pm2 logs strategy_test_hf_live --lines 100 | grep "Latency"
```

**預期輸出**:
```
📊 [Latency] tick_wait=42.3us calc=18.7us total=61.0us
📊 [Latency] tick_wait=38.1us calc=21.2us total=59.3us
```

**延遲指標解讀**:
- `tick_wait_us`: 行情等待延遲（< 100μs 為優秀）
- `factor_calc_us`: 因子計算耗時（< 50μs 為優秀）
- `total_elapsed_us`: 總端到端延遲（< 200μs 為優秀）

---

## 階段 6: 停止服務

### 6.1 優雅停止策略
```bash
pm2 stop strategy_test_hf_live
pm2 delete strategy_test_hf_live
```

### 6.2 停止所有服務
```bash
cd /app/scripts/binance_test
./run.sh stop

# 或手動停止
pm2 stop all
pm2 delete all
```

---

## 故障排除 (Troubleshooting)

### 問題 1: libsignal.so 加載失敗
**錯誤日誌**:
```
[  error  ] cannot open shared object file: libsignal.so
```

**解決方案**:
1. 驗證文件存在: `ls /app/hf-live/build/libsignal.so`
2. 驗證路徑正確: 檢查 `config.json` 中的 `signal_library_path`
3. 重新編譯: `cd /app/hf-live && make clean-build`

---

### 問題 2: on_factor 回調沒有觸發
**可能原因**:
1. 因子模組未加載
2. 模型未註冊
3. Market Data 未接收

**診斷步驟**:
```bash
# 檢查 C++ 日誌中的因子註冊
grep "Registered factor" /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live.log

# 檢查是否接收到 Market Data
grep -c "on_depth" /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live_py.log
```

---

### 問題 3: ex_order_id 為空或 "0"
**錯誤日誌**:
```
❌ [Invalid ex_order_id] Got '' for order 1234567890
```

**可能原因**:
1. Binance API Key 無效
2. 帳戶配置錯誤
3. 網絡連接問題

**解決方案**:
```bash
# 檢查 TD 日誌中的認證錯誤
tail -n 100 /app/runtime/td/binance/gz_user1/log/live/gz_user1.log | grep -i "error\|login"

# 驗證 API Key 配置
cat ~/.config/kungfu/app/runtime/config/td/binance/gz_user1.json
```

---

### 問題 4: PM2 服務啟動失敗
**現象**: `pm2 list` 顯示 `errored` 或 `stopped`

**診斷**:
```bash
# 查看錯誤日誌
pm2 logs <service_name> --err --lines 50

# 查看詳細信息
pm2 show <service_name>
```

**常見原因**:
- 埠已被占用（Master 預設 9000）
- 依賴服務未啟動（違反啟動順序）
- Python 路徑錯誤

---

## 成功標準檢查清單

運行測試後，確認以下所有項目：

- [ ] **服務啟動**: `pm2 list` 顯示 5 個服務全部 `online`
- [ ] **hf-live 加載**: C++ 日誌中出現 `Signal library loaded successfully`
- [ ] **策略初始化**: Python 日誌中出現 `🏁 [Phase 6] Pre-Start`
- [ ] **Market Data**: Python 日誌中出現持續的 `📊 [on_depth]` 輸出
- [ ] **因子計算**: Python 日誌中出現 `🤖 [LinearModel]` 輸出
- [ ] **訂單提交**: 日誌中出現 `🎉 訂單已成功提交` 且 `ex_order_id` 有效
- [ ] **訂單取消**: 30 秒後出現 `🎉 [Test Complete] Order cancelled`

---

## 關鍵文件路徑速查

### 配置文件
- 策略配置: `/app/strategies/test_hf_live/config.json`
- PM2 配置: `/app/scripts/binance_test/strategy_test_hf_live.json`
- API 配置: `~/.config/kungfu/app/runtime/config/td/binance/gz_user1.json`

### 日誌文件
- C++ 日誌: `/app/runtime/strategy/default/test_hf_live/log/live/test_hf_live.log`
- Python 日誌: `/app/runtime/strategy/default/test_hf_live/log/live/test_hf_live_py.log`
- MD 日誌: `/app/runtime/md/binance/binance/log/live/binance.log`
- TD 日誌: `/app/runtime/td/binance/gz_user1/log/live/gz_user1.log`

### 腳本
- 啟動腳本: `/app/scripts/binance_test/run.sh`
- 優雅停止: `/app/scripts/binance_test/graceful_shutdown.sh`

---

## 快速啟動命令速查

```bash
# === 完整啟動流程（一鍵複製） ===
docker exec -it godzilla-dev bash -c "
cd /app/scripts/binance_test && \
./run.sh start && \
sleep 25 && \
pm2 start strategy_test_hf_live.json && \
sleep 5 && \
pm2 logs strategy_test_hf_live
"

# === 查看所有日誌（推薦） ===
docker exec -it godzilla-dev pm2 logs

# === 只看策略日誌 ===
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live

# === 查看最近 100 行 Python 日誌 ===
docker exec godzilla-dev tail -n 100 /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live_py.log

# === 實時監控 on_factor 輸出 ===
docker exec godzilla-dev tail -f /app/runtime/strategy/default/test_hf_live/log/live/test_hf_live_py.log | grep --line-buffered "LinearModel"

# === 停止所有服務 ===
docker exec godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh stop"
```
