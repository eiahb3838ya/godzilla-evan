# scripts/helloworld

## 目的
Binance 完整交易測試環境（市場數據 + 交易執行）

## 啟動的服務
- Master (核心進程管理)
- Ledger (賬本服務)
- MD Gateway (Market Data - Binance WebSocket)
- **TD Gateway (Trade - Binance REST API)** ← 完整交易功能
- Strategy: (根據配置啟動策略)

## 使用方法
```bash
cd /app/scripts/helloworld
bash run.sh start
```

## 注意事項
**命名混淆警告**: 雖然此目錄名為 `helloworld`，但實際上這是**完整的 Binance 交易測試環境**，包含 MD + TD Gateway。這不是最簡單的入門示例。

如果只需要市場數據示例（無交易功能），請使用 `scripts/binance_test/`（是的，名稱是反的）。

## 與 binance_test 的區別
| 目錄 | MD Gateway | TD Gateway | 適合場景 |
|------|-----------|-----------|---------|
| `scripts/binance_test` | ✓ | ✗ | 入門學習，只看行情 |
| `scripts/helloworld` | ✓ | ✓ | 完整測試，下單交易 |
