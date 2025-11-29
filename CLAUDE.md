# CLAUDE.md

## 專案概述

低延遲加密貨幣交易系統，三層架構：

```
Python Strategy Layer (策略邏輯)
         ↓ pybind11
Wingchun (C++) - 策略執行、訂單管理、持倉追蹤
         ↓
Yijinjing (C++) - 事件溯源 Journal (~50μs 延遲)
         ↓
Exchange Gateways - Binance REST/WebSocket
```

**語言偏好**: 繁體中文 (zh-TW)

## 文檔優先順序

`.doc/` > `core/` 源碼 > `strategies/` 範例

**按需載入**:
- 架構細節: `.doc/modules/`
- API 契約: `.doc/contracts/`
- 操作指南: `.doc/operations/`
- 配置說明: `.doc/config/`

## 必用指令 (Docker 內執行)

```bash
# 啟動所有服務
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

# 查看狀態
docker exec godzilla-dev pm2 list

# 查看日誌
docker exec -it godzilla-dev pm2 logs

# 停止所有服務
docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all

# 進入容器
docker exec -it godzilla-dev bash

# 清除 Journal (開發用)
docker exec godzilla-dev bash -c "find ~/.config/kungfu/app/ -name '*.journal' | xargs rm -f"
```

## 啟動順序 (關鍵)

```
Master → (5s) → Ledger → (5s) → MD → (5s) → TD → (5s) → Strategy
```

## 絕對禁止

1. ❌ 在 host 執行 `python3 dev_run.py` — 必須在 Docker 內
2. ❌ 不用 PM2 管理進程 — 必須用 PM2
3. ❌ 提交 `access_key`, `secret_key`, `passphrase`
4. ❌ 錯誤的啟動順序 — Master 必須第一個啟動

## 建置指令 (Docker 內)

```bash
cd /app/core/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## 關鍵檔案位置

**資料結構** (`core/cpp/wingchun/include/kungfu/wingchun/msg.h`):
- `Order`: 666-730 行
- `Depth`: 242-302 行
- `Position`: 1000-1071 行
- `Asset`: 947-998 行

**策略執行** (`core/cpp/wingchun/src/strategy/runner.cpp`):
- 生命週期: 55-194 行
- Depth 事件: 66-76 行
- Order 路由: 124-141 行

**Python 綁定** (`core/cpp/wingchun/pybind/pybind_wingchun.cpp`):
- 枚舉綁定: 264-319 行
- Order 綁定: 516-547 行
- Context API: 719-743 行

**Binance 配置** (`core/extensions/binance/include/common.h`):
- 配置結構: 18-71 行

## 常見陷阱

| 陷阱 | 正確做法 |
|------|----------|
| `bid_price[0]` 是最差價 | `bid_price[0]` 是**最佳**買價（最高） |
| `ex_order_id` 有值 | 只有 `status=Submitted` 後才有值 |
| 回調執行超過 1ms | 所有回調必須 <1ms（單執行緒） |
| 路徑用 host 路徑 | 容器內路徑都是 `/app/` 開頭 |

## 策略開發模板

```python
class MyStrategy(Strategy):
    def pre_start(self, context):
        context.add_account("binance", "my_account")
        context.subscribe("binance", ["btcusdt"], InstrumentType.Spot, Exchange.BINANCE)

    def on_depth(self, context, depth):
        if depth.bid_price[0] > self.threshold:
            context.insert_order(symbol="btcusdt", side=Side.Buy, ...)

    def on_order(self, context, order):
        if order.status == OrderStatus.Filled:
            context.log().info(f"成交於 {order.avg_price}")

    def pre_stop(self, context):
        # 取消所有掛單
        pass
```

## 文檔更新

修改程式碼後，確認相關 `.doc/` 文檔已同步更新。
