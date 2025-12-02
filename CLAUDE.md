# CLAUDE.md - 系統元認知協議

## 系統本質

低延遲加密貨幣交易系統,三層架構:

```
Python Strategy Layer (策略邏輯)
         ↓ pybind11
Wingchun (C++) - 策略執行、訂單管理、持倉追蹤
         ↓
Yijinjing (C++) - 事件溯源 Journal (~50μs 延遲)
         ↓
Exchange Gateways - Binance REST/WebSocket
```

**核心特性**: 事件驅動、單執行緒策略、完整審計日誌

---

## 溝通協議

**語言**: 繁體中文 (zh-TW)

**文檔優先序**: `.doc/` > `core/` 源碼 > `strategies/` 範例

**導航入口**: `.doc/NAVIGATION.md` (必先讀!)

---

## 系統級鐵律 (違反=崩潰)

### 1. 容器隔離原則
❌ **絕不在 host 執行服務**
```bash
# 錯誤示範
python3 dev_run.py  # ❌ 找不到依賴!

# 正確做法
docker exec godzilla-dev pm2 start ...  # ✅
```

### 2. 進程管理原則
❌ **絕不手動啟動進程**
```bash
# 錯誤示範
nohup python3 dev_run.py &  # ❌ 無法追蹤日誌!

# 正確做法
docker exec godzilla-dev pm2 start <config>.json  # ✅
```

### 3. 啟動時序原則
**必須按順序啟動** (每步間隔 5 秒):
```
Master → (5s) → Ledger → (5s) → MD → (5s) → TD → (5s) → Strategy
```

**違反後果**: 服務無法註冊、連線失敗、事件遺失

### 4. 密鑰安全原則
❌ **絕不提交以下配置項**:
- `access_key`
- `secret_key`
- `passphrase`

**檢查方法**: `git log -S "access_key" --all` (應無結果)

---

## 文檔系統使用協議

### AI 學習規則

1. **冷啟動**:
   - 先讀 `.doc/NAVIGATION.md` 建立知識地圖
   - 理解「任務→文檔」映射關係
   - 理解文檔依賴圖 (基礎層→核心層→應用層)

2. **任務導向載入**:
   - 根據用戶意圖查詢 NAVIGATION.md 推薦文檔
   - 按推薦順序載入 2-3 個文檔 (控制在 15-20k tokens)
   - 避免一次載入超過 30k tokens

3. **程式碼定位**:
   - 需要檔案行號 → 查 `.doc/CODE_INDEX.md`
   - 需要操作指令 → 查 `.doc/operations/QUICK_START.md`
   - 需要配置說明 → 查 `.doc/config/config_usage_map.md`

4. **禁止行為**:
   - ❌ 跳過 NAVIGATION.md 直接猜測檔案路徑
   - ❌ 引用未實際讀取的文檔內容
   - ❌ 一次性全載入所有文檔 (除非真的需要全局理解)

### 快速定位鉤子

| 需求 | 文檔入口 |
|------|---------|
| **開發新策略** | NAVIGATION.md#開發新策略 |
| **除錯問題** | NAVIGATION.md#除錯Binance問題 |
| **服務部署** | NAVIGATION.md#部署與服務管理 |
| **理解架構** | NAVIGATION.md#理解事件流與架構 |
| **新增交易所** | NAVIGATION.md#新增交易所Gateway |
| **修改資料結構** | NAVIGATION.md#修改核心資料結構 |
| **操作指令** | operations/QUICK_START.md |
| **程式碼錨點** | CODE_INDEX.md |

---

## Token 預算管理

**冷啟動成本**: CLAUDE.md (本文件) + NAVIGATION.md ≈ **800 tokens**

**一般任務**: 再載入 2-3 個文檔 ≈ **15-20k tokens**

**複雜任務**: 先載入基礎層 (yijinjing + wingchun),再載入任務相關文檔 ≈ **25-35k tokens**

**極限**: 全量載入 36 個文檔 ≈ **576k tokens** (僅在必要時)

---

## 常見陷阱速查

| 陷阱 | 正確理解 | 相關文檔 |
|------|---------|---------|
| `bid_price[0]` 是最差價 | `bid_price[0]` 是**最佳買價**(最高) | CODE_INDEX.md#Depth |
| `ex_order_id` 立即有值 | 只有 `status=Submitted` 後才有值 | CODE_INDEX.md#Order |
| 回調可執行長時間運算 | 所有回調必須 <1ms (單執行緒) | NAVIGATION.md#開發新策略 |
| 容器內路徑是 `/home/...` | 容器內路徑都是 `/app/` 開頭 | operations/QUICK_START.md |
| Testnet 可執行時切換 | Testnet/Mainnet 是編譯時決定 | CODE_INDEX.md#Binance |

---

## 快速啟動 (一鍵)

```bash
# 啟動所有服務
docker exec -it godzilla-dev bash -c "cd /app/scripts/binance_test && ./run.sh start"

# 查看狀態
docker exec godzilla-dev pm2 list

# 查看日誌
docker exec -it godzilla-dev pm2 logs
```

**詳細指令**: 見 `.doc/operations/QUICK_START.md`

---

## 文檔維護

修改程式碼後,確認相關 `.doc/` 文檔已同步更新:

- 資料結構變更 → `contracts/*_object_contract.md` + `CODE_INDEX.md`
- API 變更 → `contracts/strategy_context_api.md`
- 配置變更 → `config/config_usage_map.md`
- 架構決策 → 新增 `adr/00X-decision-name.md`

**驗證工具**:
```bash
python3 .doc/operations/scripts/verify_code_refs.py  # 檢查程式碼引用
python3 .doc/operations/scripts/check_links.py       # 檢查連結完整性
```
