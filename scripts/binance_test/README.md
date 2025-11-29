# scripts/binance_test

## 目的
測試 Binance 連接的**入門級示例**（僅市場數據，無交易功能）

## 啟動的服務
- Master (核心進程管理)
- Ledger (賬本服務)
- MD Gateway (Market Data - Binance WebSocket)
- Strategy: hello (helloworld 策略)

## 配置文件
- `conf.json` - 策略配置（Binance testnet 參數）
- `strategy_hello.json` - PM2 策略進程配置

## 使用方法
```bash
cd /app/scripts/binance_test
bash run.sh start
```

## 注意事項
**命名混淆警告**: 雖然此目錄名為 `binance_test`，但實際上只啟動了 MD Gateway（市場數據），**沒有** TD Gateway（交易功能）。這是一個簡單的入門示例。

如果需要完整的交易測試環境（MD + TD），請使用 `scripts/helloworld/`（是的，名稱是反的）。
