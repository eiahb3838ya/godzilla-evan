# test_hf_live 端到端測試啟動指南

## 快速啟動

### Phase 4A: 測試基礎服務

在 **host** 執行（推薦）：
```bash
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"
docker exec godzilla-dev pm2 list
```

**期待結果**: 看到 master, ledger, md_binance, td_binance 都是 `online` 狀態

---

### Phase 4B: 測試簡單策略（無 signal library）

在 **host** 執行：
```bash
docker exec godzilla-dev pm2 start /app/scripts/test_hf_live/strategy.json
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live --lines 50
```

**期待結果**: 看到以下日誌
```
strategy_test_hf_live  | 🏁 [test_hf_live] Pre-Start
strategy_test_hf_live  | ✅ [on_depth] btcusdt bid=42000.50 ask=42001.20
```

---

## 一鍵啟動（包含策略）

在 **容器內** 執行：
```bash
docker exec -it godzilla-dev bash
cd /app/scripts/test_hf_live
./run.sh start
```

---

## 查看日誌

```bash
# 實時查看所有服務日誌
docker exec -it godzilla-dev pm2 logs

# 只查看策略日誌
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live

# 查看最後 100 行
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live --lines 100 --nostream
```

---

## 驗證數據流（漸進式）

### Phase 4A ✓ 基礎服務
```bash
docker exec godzilla-dev pm2 list
# 確認 master/ledger/md/td 都是 online
```

### Phase 4B ⏸️ Python 回調
```bash
docker exec -it godzilla-dev pm2 logs strategy_test_hf_live | grep "🏁\|✅"
```

### Phase 4C-4F ⏸️ Signal Library 集成
- 需要進一步研究 libsignal.so 加載方式
- 參考 `plan/prd_hf-live.10-e2e-testing.md` Phase 4C-4F

---

## 停止服務

```bash
docker exec -it godzilla-dev bash -c "cd /app/scripts/test_hf_live && ./run.sh stop"
```

或

```bash
docker exec godzilla-dev pm2 stop all && docker exec godzilla-dev pm2 delete all
```

---

## 故障排除

### 策略無法啟動
- 檢查 PM2 日誌: `pm2 logs strategy_test_hf_live --err`
- 確認依賴服務都已啟動: `pm2 list`

### 收不到 on_depth 回調
- 檢查 MD gateway: `pm2 logs md_binance`
- 確認 symbol 訂閱格式: `btcusdt` (小寫+底線)

### Master/Ledger 無法啟動
- 清空 journal: `find ~/.config/kungfu/app/ -name "*.journal" | xargs rm -f`
- 檢查端口占用: `netstat -tlnp`

---

## 後續步驟

- [x] Phase 4A: 基礎服務啟動測試
- [x] Phase 4B: 簡單策略測試（無 signal library）
- [ ] Phase 4C: 研究 libsignal.so 集成方式
- [ ] Phase 4D: 驗證因子層日誌
- [ ] Phase 4E: 驗證模型層日誌
- [ ] Phase 4F: 驗證 on_factor 回調

詳細計劃見: `plan/prd_hf-live.10-e2e-testing.md`
