# 快速啟動指令參考

本文件集中所有常用 Docker 和 PM2 指令,供快速複製貼上使用。

**重要**: 所有服務必須在 Docker 容器內執行,絕不在 host 直接運行!

---

## 一、服務管理 (PM2)

### 🚀 啟動所有服務 (一鍵)

```bash
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"
```

**說明**: 自動按正確順序啟動 Master → Ledger → MD → TD (每步間隔 5 秒)

---

### 📊 查看服務狀態

```bash
# 列出所有服務
docker exec godzilla-dev pm2 list

# 查看即時日誌 (所有服務)
docker exec -it godzilla-dev pm2 logs

# 查看特定服務日誌
docker exec -it godzilla-dev pm2 logs master
docker exec -it godzilla-dev pm2 logs ledger
docker exec -it godzilla-dev pm2 logs md_binance
docker exec -it godzilla-dev pm2 logs td_binance

# 即時監控 (CPU, Memory, 日誌)
docker exec -it godzilla-dev pm2 monit
```

---

### 🛑 停止服務

```bash
# 停止並刪除所有服務
docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all

# 停止特定服務
docker exec godzilla-dev pm2 stop master
docker exec godzilla-dev pm2 stop ledger

# 重啟特定服務
docker exec godzilla-dev pm2 restart td_binance
```

---

### 🗑️ 清除 Journal (開發用)

```bash
# ⚠️ 警告: 會刪除所有歷史事件記錄,僅用於開發環境!
docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"
```

**用途**:
- 測試時清除舊數據
- 解決 Journal 損壞問題
- 重置開發環境

**不適用於**: 生產環境 (會丟失審計日誌)

---

## 二、手動啟動 (分步驟)

### 方法 1: Host 執行 (推薦)

```bash
# 1. 啟動 Master (等待 5 秒)
docker exec godzilla-dev pm2 start /app/scripts/binance_test/master.json
sleep 5

# 2. 啟動 Ledger (等待 5 秒)
docker exec godzilla-dev pm2 start /app/scripts/binance_test/ledger.json
sleep 5

# 3. 啟動 MD Gateway (等待 5 秒)
docker exec godzilla-dev pm2 start /app/scripts/binance_test/md_binance.json
sleep 5

# 4. 啟動 TD Gateway (等待 5 秒)
docker exec godzilla-dev pm2 start /app/scripts/binance_test/td_binance.json
sleep 5

# 5. 啟動策略 (範例)
docker exec godzilla-dev pm2 start /app/scripts/demo_future/strategy_demo_future.json
```

---

### 方法 2: 容器內執行 (學習用)

```bash
# 進入容器
docker exec -it godzilla-dev bash

# 以下指令在容器內執行
cd /app/scripts/binance_test

pm2 start master.json
sleep 5

pm2 start ledger.json
sleep 5

pm2 start md_binance.json
sleep 5

pm2 start td_binance.json
sleep 5

# 啟動你的策略
cd /app/scripts/demo_future
pm2 start strategy_demo_future.json
```

---

## 三、容器操作

### 進入容器 Shell

```bash
docker exec -it godzilla-dev bash
```

**用途**:
- 手動執行指令
- 查看檔案系統
- 除錯環境問題

---

### 容器生命週期

```bash
# 查看容器狀態
docker ps | grep godzilla-dev

# 啟動容器 (如果未運行)
docker-compose up -d

# 停止容器
docker-compose stop

# 重啟容器
docker-compose restart

# 查看容器日誌
docker-compose logs -f app
```

---

## 四、建置與編譯

### C++ 核心編譯

```bash
# 方法 1: Host 執行
docker exec -it godzilla-dev bash -c "cd /app/core/build && make -j\$(nproc)"

# 方法 2: 容器內執行
docker exec -it godzilla-dev bash
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Build Types**:
- `Release`: 生產環境 (-O3 最佳化)
- `Debug`: 開發除錯 (-O0 -g)
- `RelWithDebInfo`: 效能分析 (-O3 -g)

---

### 清除重建

```bash
docker exec -it godzilla-dev bash -c "cd /app/core/build && rm -rf * && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j\$(nproc)"
```

**用途**:
- 切換 Build Type
- 解決編譯錯誤
- 重新生成 Python bindings

---

### Python Bindings 檢查

```bash
# 確認 bindings 已生成
docker exec godzilla-dev ls -la /app/core/build/kfc/python/

# 測試 import
docker exec godzilla-dev python3 -c "from kungfu.wingchun import Strategy; print('OK')"
```

---

## 五、配置管理

### 查看配置檔

```bash
# 容器內配置路徑: ~/.config/kungfu/app/runtime/config/

# 查看 TD 配置 (Binance)
docker exec godzilla-dev cat ~/.config/kungfu/app/runtime/config/td/binance/<account_name>.json

# 查看 MD 配置
docker exec godzilla-dev cat ~/.config/kungfu/app/runtime/config/md/binance/config.json

# 列出所有配置檔
docker exec godzilla-dev find ~/.config/kungfu/app/runtime/config/ -name "*.json"
```

---

### 編輯配置檔

```bash
# ⚠️ 不推薦在容器內編輯,應在 host 編輯後重啟服務

# Host 路徑 (假設有 volume mount)
# 編輯 host 上的檔案:
# ~/.config/kungfu/app/runtime/config/td/binance/<account>.json

# 或使用 docker cp 複製檔案
docker cp <local_file> godzilla-dev:/root/.config/kungfu/app/runtime/config/td/binance/<account>.json

# 重啟相關服務
docker exec godzilla-dev pm2 restart td_binance
```

---

## 六、日誌查詢

### PM2 日誌

```bash
# 即時日誌 (Ctrl+C 退出)
docker exec -it godzilla-dev pm2 logs

# 查看歷史日誌 (最後 100 行)
docker exec godzilla-dev pm2 logs --lines 100

# 僅顯示錯誤
docker exec -it godzilla-dev pm2 logs --err

# 清空日誌
docker exec godzilla-dev pm2 flush
```

---

### Runtime 日誌

```bash
# Journal 日誌位置
docker exec godzilla-dev ls -la ~/.config/kungfu/app/runtime/journal/

# TD Runtime 日誌 (Binance)
docker exec godzilla-dev tail -n 50 ~/.config/kungfu/app/runtime/log/td/binance/<account>/runtime/<date>.log

# MD Runtime 日誌
docker exec godzilla-dev tail -n 50 ~/.config/kungfu/app/runtime/log/md/binance/runtime/<date>.log

# 策略日誌
docker exec godzilla-dev tail -n 50 ~/.config/kungfu/app/runtime/log/strategy/<strategy_name>/runtime/<date>.log
```

---

## 七、除錯與診斷

### 環境診斷

```bash
# 完整診斷 (檢查 Docker, PM2, 配置)
docker exec godzilla-dev bash /app/.doc/operations/scripts/diagnostic.sh

# 驗證 CLI 指令
docker exec godzilla-dev bash /app/.doc/operations/scripts/verify-commands.sh
```

---

### 網路問題

```bash
# 測試 Binance 連線
docker exec godzilla-dev curl -I https://api.binance.com/api/v3/ping
docker exec godzilla-dev curl -I https://testnet.binance.vision/api/v3/ping

# DNS 檢查
docker exec godzilla-dev nslookup api.binance.com

# 修復 DNS 問題
bash .doc/operations/scripts/setup-docker-dns.sh
```

---

### 進程檢查

```bash
# 查看進程樹
docker exec godzilla-dev ps aux | grep kungfu

# 檢查埠占用
docker exec godzilla-dev netstat -tlnp | grep LISTEN

# 檢查 Journal 鎖
docker exec godzilla-dev lsof | grep journal
```

---

## 八、常見場景速查

| 我想... | 指令 |
|--------|------|
| **快速啟動所有服務** | `docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"` |
| **查看服務狀態** | `docker exec godzilla-dev pm2 list` |
| **查看即時日誌** | `docker exec -it godzilla-dev pm2 logs` |
| **停止所有服務** | `docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all` |
| **進入容器** | `docker exec -it godzilla-dev bash` |
| **重新編譯** | `docker exec -it godzilla-dev bash -c "cd /app/core/build && make -j\$(nproc)"` |
| **清除 Journal** | `docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"` |
| **測試 Binance 連線** | `docker exec godzilla-dev curl -I https://testnet.binance.vision/api/v3/ping` |

---

## 九、安全提醒

### ❌ 絕對禁止

```bash
# ❌ 在 host 直接運行 (會找不到依賴)
python3 core/python/dev_run.py  # 錯誤!

# ❌ 不用 PM2 管理進程 (難以追蹤日誌)
nohup python3 dev_run.py &  # 錯誤!

# ❌ 錯誤的啟動順序 (會導致連線失敗)
pm2 start td_binance.json  # 錯誤! Master 必須先啟動
```

### ✅ 正確做法

```bash
# ✅ 一律通過 docker exec 執行
docker exec godzilla-dev pm2 start /app/scripts/binance_test/master.json

# ✅ 使用 run.sh 自動處理啟動順序
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"
```

---

## 十、進階操作

### 創建新策略的 PM2 配置

```json
{
  "apps": [{
    "name": "strategy_<your_name>",
    "cwd": "/app",
    "script": "/app/core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info strategy -n <your_name> -p /app/strategies/<your_name>/<your_name>.py -c /app/strategies/<your_name>/config.json",
    "watch": false,
    "env": {
      "KF_HOME": "/app/runtime"
    }
  }]
}
```

**啟動**:
```bash
docker exec godzilla-dev pm2 start /app/scripts/<your_name>/strategy_<your_name>.json
```

---

### 備份與還原配置

```bash
# 備份配置
docker exec godzilla-dev tar -czf /tmp/config_backup.tar.gz -C ~/.config/kungfu/app/runtime config/
docker cp godzilla-dev:/tmp/config_backup.tar.gz ./config_backup_$(date +%Y%m%d).tar.gz

# 還原配置
docker cp ./config_backup_20251201.tar.gz godzilla-dev:/tmp/
docker exec godzilla-dev tar -xzf /tmp/config_backup_20251201.tar.gz -C ~/.config/kungfu/app/runtime/
docker exec godzilla-dev pm2 restart all
```

---

## 參考資料

- 詳細操作指南: [operations/pm2_startup_guide.md](pm2_startup_guide.md)
- CLI 工具說明: [operations/cli_operations_guide.md](cli_operations_guide.md)
- 除錯流程: [operations/debugging_guide.md](debugging_guide.md)
- 配置管理: [config/config_usage_map.md](../config/config_usage_map.md)
