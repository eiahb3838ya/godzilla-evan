# Demo Future 策略

## 概述

這是一個 Binance Futures（期貨）交易策略示範，展示如何使用 Godzilla-Evan 交易系統進行期貨合約交易。

## 功能特性

- **市場類型**：Binance 永續合約（FFuture）
- **訂閱數據**：指數價格（Index Price）
- **交易邏輯**：當沒有活躍訂單時，下限價買單
- **訂單管理**：追蹤活躍訂單狀態，查詢未確認訂單

## 策略回調

### `pre_start(context)`
初始化階段，執行以下操作：
- 添加交易帳戶
- 設置帳戶資金限制（base_coin 和 quote_coin）
- 訂閱指數價格數據

### `on_index_price(context, price)`
接收指數價格更新：
- 記錄指數價格
- 查詢當前持倉

### `on_depth(context, depth)`
接收深度數據更新（如果訂閱）：
- 檢查活躍訂單數量
- 無活躍訂單時下單
- 有活躍訂單時查詢訂單狀態

### `on_order(context, order)`
訂單狀態變化回調：
- 記錄訂單狀態
- 可擴展訂單管理邏輯

### `on_position(context, position)`
持倉更新回調：
- 記錄持倉變化

## 配置說明

### `config.json` 結構

```json
{
  "name": "demo_future",          // 策略名稱
  "md_source": "binance",         // 市場數據源
  "td_source": "binance",         // 交易數據源
  "symbol": "btcusdt",            // 交易合約符號
  "account": "gz_user1",          // 交易帳戶名稱
  "base_coin": "usdt",            // 基礎貨幣（保證金）
  "quote_coin": "btc",            // 報價貨幣
  "base_limit": 10000,            // 基礎貨幣限額（USDT）
  "quote_limit": 1                // 報價貨幣限額（BTC）
}
```

### 配置項說明

- **symbol**：必須是有效的 Binance Futures 合約符號（小寫）
- **account**：必須與 TD 配置中的帳戶名稱一致
- **base_coin/quote_coin**：用於設置帳戶資金限制
- **base_limit/quote_limit**：Ledger 用於餘額檢查的虛擬限額

## 啟動方式

### 使用 PM2 腳本（推薦）

```bash
cd scripts/demo_future
./run.sh start
```

### 手動啟動

```bash
# 1. 啟動 Master
python3 core/python/dev_run.py -l info master
sleep 5

# 2. 啟動 Ledger
python3 core/python/dev_run.py -l info ledger
sleep 5

# 3. 啟動 MD Gateway
python3 core/python/dev_run.py -l trace md -s binance
sleep 5

# 4. 啟動 TD Gateway
python3 core/python/dev_run.py -l info td -s binance -a gz_user1
sleep 5

# 5. 啟動策略
python3 core/python/dev_run.py -l info strategy \
  -n demo_future \
  -p strategies/demo_future/demo_future.py \
  -c strategies/demo_future/config.json
```

## 檢查執行狀態

```bash
# 查看所有服務狀態
pm2 list

# 查看策略日誌
pm2 logs strategy:demo_future --lines 50

# 查看 TD Gateway 日誌（查看訂單執行）
pm2 logs td_binance:gz_user1 --lines 50

# 查看 MD Gateway 日誌（查看市場數據）
pm2 logs md_binance --lines 50
```

## 停止策略

```bash
cd scripts/demo_future
./run.sh stop
```

## 重要事項

### ⚠️ Futures 配置要求

#### 配置存儲位置

**配置存儲在 SQLite 數據庫中**（不是 JSON 文件）：

```
/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db
```

**表名**：`account_config`

#### 檢查當前配置

使用以下 Python 腳本查看配置：

```bash
python3 << 'EOF'
import sqlite3
import json

db_path = '/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_gz_user1';")
config_json = cursor.fetchone()[0]
config = json.loads(config_json)

print("=== 當前 Binance 配置 ===")
print(f"enable_spot: {config.get('enable_spot', True)}")
print(f"enable_futures: {config.get('enable_futures', True)}")

conn.close()
EOF
```

**預期輸出**（Futures Testnet 的正確配置）：
```
=== 當前 Binance 配置 ===
enable_spot: False
enable_futures: True
```

#### 為什麼 `enable_spot=False` 是正確的？

**重要**：Binance **Futures Testnet** 和 **Spot Testnet** 是完全分開的系統，使用不同的 API 金鑰。

- Futures Testnet API 金鑰 **無法** 訪問 Spot 端點
- 如果 `enable_spot=True`，系統會嘗試用 Futures API 金鑰訪問 Spot 端點 → `-2015` 錯誤
- **正確配置**：
  - `enable_spot: false`（避免 -2015 錯誤）
  - `enable_futures: true`（啟用 Futures 交易）

**參考**：[.doc/40_config/config_usage_map.md](../../.doc/40_config/config_usage_map.md#enable_spot-optional)

#### Testnet vs Mainnet

端點配置**硬編碼在源碼中**（不可通過配置修改）：

- **測試環境**（當前）：
  - Futures REST: `testnet.binancefuture.com`
  - Futures WSS: `stream.binancefuture.com`

- **生產環境**（需修改源碼並重新編譯）：
  - Futures REST: `fapi.binance.com`
  - Futures WSS: `fstream.binance.com`

#### 保證金模式

- 期貨交易需要保證金，確保帳戶有足夠的 USDT 餘額
- 預設槓桿倍數由交易所配置決定

### 🔒 安全注意事項

- 永遠不要提交包含 API 密鑰的配置檔案
- 測試階段使用 Binance Testnet
- 小額測試後再投入實際資金

## 故障排除

### 訂單被拒絕（-2015 錯誤）

**症狀**：TD Gateway 日誌顯示 `-2015` 錯誤

**可能原因 1**：使用 Futures API 金鑰，但 `enable_spot=True`

**解決方案**：
```bash
# 檢查配置
python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db')
cursor = conn.cursor()
cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_gz_user1';")
config = json.loads(cursor.fetchone()[0])
print(f"enable_spot: {config.get('enable_spot')}, enable_futures: {config.get('enable_futures')}")
conn.close()
EOF

# 如果 enable_spot=True，需要改為 False：
python3 << 'EOF'
import sqlite3, json
conn = sqlite3.connect('/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db')
cursor = conn.cursor()
cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_gz_user1';")
config = json.loads(cursor.fetchone()[0])
config['enable_spot'] = False
cursor.execute("UPDATE account_config SET config = ? WHERE account_id = 'binance_gz_user1'", (json.dumps(config),))
conn.commit()
conn.close()
print("已設置 enable_spot=False")
EOF

# 重啟 TD Gateway
pm2 restart td_binance:gz_user1
```

**可能原因 2**：`enable_futures=False`

**解決方案**：將 `enable_futures` 設為 `True`（參考上述腳本）

### 無法連接到 Futures WebSocket

**原因**：MD 配置中缺少 Futures WebSocket 端點

**解決方案**：檢查 `futures_wss_host` 配置

### 策略無法接收市場數據

**原因**：啟動順序不正確或等待時間不足

**解決方案**：按照 Master → Ledger → MD → TD → Strategy 順序啟動，每步等待 5 秒

## 參考文檔

- 策略開發框架：`.doc/10_modules/strategy_framework.md`
- Binance 擴展：`.doc/10_modules/binance_extension.md`
- 訂單生命週期：`.doc/20_interactions/order_lifecycle_flow.md`
- Context API：`.doc/30_contracts/strategy_context_api.md`
- CLI 操作指南：`.doc/90_operations/cli_operations_guide.md`

## 版本歷史

- **2025-11-21**：初始版本，建立 demo_future 策略結構
