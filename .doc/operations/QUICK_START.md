# 快速啟動指令參考

本文件集中所有常用 Docker 和 PM2 指令,供快速複製貼上使用。

**重要**: 所有服務必須在 Docker 容器內執行!

---

## 🚀 一鍵啟動/停止

```bash
# 啟動所有服務 (自動順序: Master → Ledger → MD → TD)
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

# 停止所有服務
docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all
```

---

## 📊 服務監控

```bash
# 列出所有服務
docker exec godzilla-dev pm2 list

# 查看即時日誌
docker exec -it godzilla-dev pm2 logs          # 所有服務
docker exec -it godzilla-dev pm2 logs master   # 特定服務

# CPU/Memory 監控
docker exec -it godzilla-dev pm2 monit
```

---

## 🔧 手動啟動 (分步驟)

```bash
# 從 host 執行 (推薦)
docker exec godzilla-dev pm2 start /app/scripts/binance_test/master.json && sleep 5
docker exec godzilla-dev pm2 start /app/scripts/binance_test/ledger.json && sleep 5
docker exec godzilla-dev pm2 start /app/scripts/binance_test/md_binance.json && sleep 5
docker exec godzilla-dev pm2 start /app/scripts/binance_test/td_binance.json && sleep 5

# 或在容器內執行
docker exec -it godzilla-dev bash
cd /app/scripts/binance_test
pm2 start master.json && sleep 5
pm2 start ledger.json && sleep 5
pm2 start md_binance.json && sleep 5
pm2 start td_binance.json
```

**啟動順序**: Master → Ledger → MD → TD → Strategy (每步間隔 5秒)

---

## 🗑️ 清除 Journal (開發用)

```bash
# ⚠️ 警告: 刪除所有歷史事件記錄,僅用於開發環境!
docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"
```

---

## 🐳 容器操作

```bash
# 進入容器
docker exec -it godzilla-dev bash

# 容器生命週期
docker-compose up -d        # 啟動
docker-compose stop         # 停止
docker-compose restart      # 重啟
docker ps | grep godzilla   # 狀態
```

---

## 🔨 編譯與構建

```bash
# C++ 核心編譯 (Release 模式)
docker exec -it godzilla-dev bash -c "cd /app/core/build && make -j\$(nproc)"

# 清除重建 (切換 Build Type 或解決編譯錯誤)
docker exec -it godzilla-dev bash -c "cd /app/core/build && rm -rf * && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j\$(nproc)"

# 驗證 Python bindings
docker exec godzilla-dev python3 -c "from kungfu.wingchun import Strategy; print('OK')"
```

**Build Types**: `Release` (生產), `Debug` (除錯), `RelWithDebInfo` (效能分析)

---

## ⚙️ 配置管理

```bash
# 查看配置檔 (容器內路徑: ~/.config/kungfu/app/runtime/config/)
docker exec godzilla-dev cat ~/.config/kungfu/app/runtime/config/td/binance/<account>.json

# 列出所有配置
docker exec godzilla-dev find ~/.config/kungfu/app/runtime/config/ -name "*.json"
```

**編輯配置**: 在 host 編輯後重啟服務,或使用 `docker cp` 複製檔案

---

## 📝 日誌查詢

```bash
# PM2 日誌
docker exec -it godzilla-dev pm2 logs --lines 100    # 最後 100 行
docker exec -it godzilla-dev pm2 logs --err          # 僅錯誤
docker exec godzilla-dev pm2 flush                    # 清空日誌

# Runtime 日誌
docker exec godzilla-dev tail -n 50 ~/.config/kungfu/app/runtime/log/td/binance/<account>/runtime/<date>.log
docker exec godzilla-dev tail -n 50 ~/.config/kungfu/app/runtime/log/strategy/<strategy>/runtime/<date>.log
```

---

## 🆕 創建新策略

**PM2 配置範本** (`scripts/<name>/strategy_<name>.json`):
```json
{
  "apps": [{
    "name": "strategy_<name>",
    "cwd": "/app",
    "script": "/app/core/python/dev_run.py",
    "exec_interpreter": "python3",
    "args": "-l info strategy -n <name> -p /app/strategies/<name>/<name>.py -c /app/strategies/<name>/config.json",
    "watch": false,
    "env": {"KF_HOME": "/app/runtime"}
  }]
}
```

**啟動**:
```bash
docker exec godzilla-dev pm2 start /app/scripts/<name>/strategy_<name>.json
```

---

## 🔍 除錯診斷

```bash
# 測試 Binance 連線
docker exec godzilla-dev curl -I https://testnet.binance.vision/api/v3/ping

# 檢查進程
docker exec godzilla-dev ps aux | grep kungfu

# 檢查埠占用
docker exec godzilla-dev netstat -tlnp | grep LISTEN

# DNS 檢查
docker exec godzilla-dev nslookup api.binance.com
```

---

## 🎯 常見場景速查

| 我想... | 指令 |
|--------|------|
| 快速啟動 | `docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"` |
| 查看狀態 | `docker exec godzilla-dev pm2 list` |
| 查看日誌 | `docker exec -it godzilla-dev pm2 logs` |
| 停止服務 | `docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all` |
| 進入容器 | `docker exec -it godzilla-dev bash` |
| 重新編譯 | `docker exec -it godzilla-dev bash -c "cd /app/core/build && make -j\$(nproc)"` |
| 清除 Journal | `docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"` |

---

## ⚠️ 安全提醒

### ❌ 絕對禁止
- 在 host 直接運行 `python3 dev_run.py` (找不到依賴)
- 不用 PM2 管理進程 (難以追蹤日誌)
- 錯誤的啟動順序 (TD 必須在 Master 之後)

### ✅ 正確做法
- 一律通過 `docker exec` 執行
- 使用 `run.sh` 自動處理啟動順序
- 使用 PM2 管理所有進程

---

## 📚 延伸閱讀

- [pm2_startup_guide.md](pm2_startup_guide.md) - PM2 完整操作指南
- [debugging_guide.md](debugging_guide.md) - 除錯診斷流程
- [TESTNET.md](TESTNET.md) - 測試網設定完整指南
- [CONFIG_REFERENCE.md](../config/CONFIG_REFERENCE.md) - 配置管理

---

**更新時間**: 2025-12-01
