# Demo Future 啟動腳本

## 概述

這個目錄包含使用 PM2 啟動和管理 `demo_future` Futures 交易策略的所有必要腳本和配置檔案。

## 檔案說明

### 配置檔案（PM2 JSON）

| 檔案 | 用途 | 服務名稱 |
|------|------|----------|
| `master.json` | Master 服務（服務註冊中心） | `master` |
| `ledger.json` | Ledger 服務（帳本/持倉追蹤） | `ledger` |
| `md_binance.json` | Market Data 閘道器（Binance WebSocket） | `md_binance` |
| `td_binance.json` | Trading 閘道器（Binance REST API） | `td_binance:gz_user1` |
| `strategy.json` | Demo Future 策略執行器 | `strategy:demo_future` |

### 腳本檔案

- **`run.sh`**：一鍵啟動/停止所有服務的 Shell 腳本
- **`README.md`**：本文檔

## 快速啟動

### 啟動所有服務

```bash
cd /home/huyifan/projects/godzilla-evan/scripts/demo_future
./run.sh start
```

**啟動順序**：
1. Master（等待 5 秒）
2. Ledger（等待 5 秒）
3. Market Data Gateway（等待 5 秒）
4. Trading Gateway（等待 5 秒）
5. Demo Future Strategy（等待 2 秒）

**總啟動時間**：約 22 秒

### 停止所有服務

```bash
./run.sh stop
```

這會優雅地關閉所有服務（發送 SIGINT 信號給 Master）。

## PM2 管理命令

### 查看服務狀態

```bash
# 列出所有服務
pm2 list

# 查看詳細資訊
pm2 show strategy:demo_future
```

### 查看日誌

```bash
# 即時查看所有日誌
pm2 logs

# 查看特定服務日誌
pm2 logs strategy:demo_future
pm2 logs td_binance:gz_user1
pm2 logs md_binance

# 查看最後 100 行
pm2 logs strategy:demo_future --lines 100

# 僅查看錯誤日誌
pm2 logs strategy:demo_future --err
```

### 重啟服務

```bash
# 重啟單一服務
pm2 restart strategy:demo_future

# 重啟所有服務
pm2 restart all
```

### 停止服務

```bash
# 停止單一服務
pm2 stop strategy:demo_future

# 停止所有服務
pm2 stop all

# 從 PM2 刪除服務
pm2 delete strategy:demo_future
```

## 服務啟動參數說明

### Master
```
-l info master
```
- 日誌級別：`info`
- 服務類型：`master`
- 監聽端口：9000

### Ledger
```
-l info ledger
```
- 日誌級別：`info`
- 服務類型：`ledger`

### Market Data Gateway
```
-l trace md -s binance
```
- 日誌級別：`trace`（顯示所有 WebSocket 訊息）
- 服務類型：`md`（Market Data）
- 數據源：`binance`

### Trading Gateway
```
-l info td -s binance -a gz_user1
```
- 日誌級別：`info`
- 服務類型：`td`（Trading）
- 數據源：`binance`
- 帳戶名稱：`gz_user1`

### Strategy
```
-l info strategy -n demo_future -p strategies/demo_future/demo_future.py -c strategies/demo_future/config.json
```
- 日誌級別：`info`
- 服務類型：`strategy`
- 策略名稱：`demo_future`
- 策略路徑：`strategies/demo_future/demo_future.py`
- 配置檔案：`strategies/demo_future/config.json`

## 環境變數

### KF_HOME

策略配置中設定 `KF_HOME=/app/runtime`，這會將所有運行時資料（Journal、日誌、配置）存放在：
- **Docker 內部**：`/app/runtime/`
- **Host 映射**：根據 `docker-compose.yml` 的 volume 配置

### CLEAR_JOURNAL（開發模式）

如需在啟動時清除舊的 Journal 檔案，可以在 JSON 配置中添加：
```json
"env": {
  "CLEAR_JOURNAL": "1",
  "KF_HOME": "/app/runtime"
}
```

⚠️ **警告**：`CLEAR_JOURNAL=1` 會刪除所有歷史交易數據，僅用於開發測試！

## 故障排除

### 1. 服務無法啟動

**症狀**：PM2 顯示服務狀態為 `errored` 或 `stopped`

**排查步驟**：
```bash
# 查看錯誤日誌
pm2 logs <service-name> --err --lines 50

# 常見問題：
# - Master 未運行（先啟動 Master）
# - 端口被佔用（檢查 9000 端口）
# - Python 導入錯誤（檢查 PYTHONPATH）
```

### 2. 策略無法接收市場數據

**症狀**：策略日誌中沒有 `on_depth` 或 `on_index_price` 回調

**排查步驟**：
```bash
# 1. 檢查 MD Gateway 是否連接成功
pm2 logs md_binance | grep -i "connect"

# 2. 檢查訂閱是否成功
pm2 logs md_binance | grep -i "subscribe"

# 3. 檢查策略是否正確訂閱
pm2 logs strategy:demo_future | grep -i "subscribe"
```

**可能原因**：
- MD Gateway 未連接到 Binance WebSocket
- 策略中 `subscribe_index_price` 的符號錯誤
- Binance testnet API 限流

### 3. 訂單被拒絕（-2015 錯誤）

**症狀**：TD Gateway 日誌顯示 `-2015` 錯誤

**可能原因**：

1. **使用 Futures API 金鑰，但 `enable_spot=True`**
   - Futures Testnet API 金鑰無法訪問 Spot 端點
   - 系統嘗試用 Futures 金鑰訪問 Spot → `-2015` 錯誤

2. **`enable_futures=False`**
   - Futures 功能未啟用

**檢查配置**（配置存儲在 SQLite 數據庫中）：

```bash
# 查看當前配置
python3 << 'EOF'
import sqlite3, json

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

**正確配置**（Futures Testnet）：
```
enable_spot: False
enable_futures: True
```

**如果需要修改配置**：
```bash
# 設置 enable_spot=False, enable_futures=True
python3 << 'EOF'
import sqlite3, json

db_path = '/home/huyifan/projects/godzilla-evan/runtime/system/etc/kungfu/db/live/accounts.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT config FROM account_config WHERE account_id = 'binance_gz_user1';")
config = json.loads(cursor.fetchone()[0])

config['enable_spot'] = False
config['enable_futures'] = True

cursor.execute("UPDATE account_config SET config = ? WHERE account_id = 'binance_gz_user1'", (json.dumps(config),))
conn.commit()
conn.close()

print("配置已更新")
EOF

# 重啟 TD Gateway 使配置生效
pm2 restart td_binance:gz_user1
```

### 4. WebSocket 持續斷線

**症狀**：MD Gateway 日誌顯示 `Connection closed` 反覆出現

**可能原因**：
- 網路不穩定
- Binance testnet 維護
- 訂閱過多符號導致限流

**解決方案**：
```bash
# 重啟 MD Gateway
pm2 restart md_binance

# 檢查 Binance 服務狀態
# https://testnet.binance.vision/
```

### 5. 記憶體使用過高

**症狀**：PM2 顯示某個服務記憶體 >500 MB

**解決方案**：
```bash
# 查看詳細資訊
pm2 show <service-name>

# 重啟服務清除記憶體
pm2 restart <service-name>
```

## 生產環境配置建議

### 關閉 Watch 模式

開發環境：
```json
"watch": "true"  // 檔案變更時自動重啟
```

生產環境：
```json
"watch": "false"  // 明確重啟
```

### 移除 CLEAR_JOURNAL

開發環境：
```bash
# run.sh 中包含
find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f
```

生產環境：
```bash
# 註解掉此行，保留歷史數據
# find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f
```

### 調整日誌級別

開發環境：
- MD Gateway：`-l trace`（詳細 WebSocket 訊息）
- 其他服務：`-l info`

生產環境：
- 所有服務：`-l info` 或 `-l warning`
- 減少日誌輸出，降低 I/O 開銷

### 配置日誌輪轉

```bash
# 安裝 PM2 日誌輪轉模組
pm2 install pm2-logrotate

# 配置輪轉參數
pm2 set pm2-logrotate:max_size 10M     # 單檔最大 10MB
pm2 set pm2-logrotate:retain 7         # 保留 7 天
pm2 set pm2-logrotate:compress true    # 壓縮舊日誌
```

## 目錄結構

```
scripts/demo_future/
├── master.json           # Master 服務配置
├── ledger.json           # Ledger 服務配置
├── md_binance.json       # Market Data 閘道器配置
├── td_binance.json       # Trading 閘道器配置
├── strategy.json         # Demo Future 策略配置
├── run.sh                # 啟動/停止腳本
└── README.md             # 本文檔
```

## 相關文檔

- 策略文檔：`../../strategies/demo_future/README.md`
- PM2 啟動指南：`../../.doc/90_operations/pm2_startup_guide.md`
- CLI 操作指南：`../../.doc/90_operations/cli_operations_guide.md`
- Binance 擴展：`../../.doc/10_modules/binance_extension.md`

## 版本歷史

- **2025-11-21**：初始版本，建立 demo_future 啟動腳本
