# .doc 文檔系統參考

## 目錄結構

```
.doc/
├── REFERENCE.md      # 本文件 - 文檔系統快速參考
├── modules/          # 模組說明（yijinjing, wingchun, binance）
├── contracts/        # API 契約（order, depth, context_api）
├── operations/       # 操作指南（pm2, cli, debugging）
├── config/           # 配置說明（config_map, dangerous_keys）
├── adr/              # 架構決策記錄
└── archive/          # 大檔案、教學文檔（不自動載入）
```

## 按需載入策略

| 任務類型 | 載入文檔 |
|----------|----------|
| 開發新策略 | `modules/strategy_framework.md`, `contracts/context_api.md` |
| 除錯問題 | `operations/debugging.md`, `config/config_map.md` |
| 部署服務 | `operations/pm2_guide.md`, `operations/cli_guide.md` |
| 新增交易所 | `modules/gateway_architecture.md`, `modules/binance.md` |
| 架構決策 | `adr/*.md` |

## 核心模組摘要

### Yijinjing (易筋經) - 事件溯源
- **職責**: 事件記錄、訊息傳遞、時間旅行
- **延遲**: ~50-200μs
- **詳細**: `modules/yijinjing.md`

### Wingchun (詠春) - 交易框架
- **職責**: 策略執行、訂單管理、持倉追蹤
- **回調**: `pre_start()` → `on_depth()` / `on_order()` → `pre_stop()`
- **詳細**: `modules/wingchun.md`, `modules/strategy_framework.md`

### Binance Extension - 交易所連接器
- **職責**: REST API + WebSocket 實作
- **支援**: Spot + Futures (Testnet/Mainnet)
- **詳細**: `modules/binance.md`

## API 契約摘要

### Order 物件
```
欄位: order_id, status, volume, volume_traded, avg_price, ex_order_id
不變量: volume_traded <= volume
陷阱: ex_order_id 在 Submitted 後才有值
```

### Depth 物件
```
欄位: bid_price[10], ask_price[10], bid_volume[10], ask_volume[10]
不變量: bid_price[0] > bid_price[1] (降序), ask_price[0] < ask_price[1] (升序)
陷阱: bid_price[0] 是最佳買價（最高），不是最差
```

### Context API
```python
# 帳戶管理
context.add_account(source, account)

# 訂閱市場數據
context.subscribe(source, symbols, instrument_type, exchange)

# 下單
context.insert_order(symbol, side, price, volume, ...)

# 取消訂單
context.cancel_order(order_id)

# 狀態管理
context.set_object(key, value)
context.get_object(key)
```

## 配置要點

**配置位置**: `~/.config/kungfu/app/runtime/config/`

**危險配置項** (絕不提交):
- `access_key` - API 金鑰
- `secret_key` - API 密鑰
- `passphrase` - API 密碼短語

**Testnet vs Mainnet**: 硬編碼在 `core/extensions/binance/include/common.h:18-71`

## 操作指令快速參考

```bash
# PM2 狀態
docker exec godzilla-dev pm2 list
docker exec godzilla-dev pm2 logs [service_name]
docker exec godzilla-dev pm2 monit

# 服務啟動
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

# 服務停止
docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all

# 建置
docker exec -it godzilla-dev bash -c "cd /app/core/build && make -j\$(nproc)"
```

## 程式碼錨點

**資料結構** (`msg.h`):
- Order: 666-730
- Depth: 242-302
- Position: 1000-1071
- Asset: 947-998

**策略執行** (`runner.cpp`):
- 生命週期: 55-194
- Depth 事件: 66-76
- Order 路由: 124-141

**Python 綁定** (`pybind_wingchun.cpp`):
- 枚舉: 264-319
- Order: 516-547
- Context: 719-743

## 維護指南

修改程式碼後：
1. 更新相關 `modules/*.md` 文檔
2. 如有 API 變更，更新 `contracts/*.md`
3. 如有配置變更，更新 `config/*.md`
4. 重大架構決策記錄到 `adr/`
