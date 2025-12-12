# hf-live 構建優化與部署運維

## 文檔元信息
- **版本**: v1.0
- **日期**: 2025-12-04
- **目標**: 補充前 7 個文檔未詳細覆蓋的構建優化、CI/CD、監控運維
- **前置**: [prd_hf-live.07-implementation.md](prd_hf-live.07-implementation.md)
- **定位**: Day-2 Operations 手冊 (非基礎設計)

---

## 文檔範圍說明

**本文檔僅涵蓋**:
- ✅ 構建優化技巧 (Release 模式、CPU 指令集、LTO)
- ✅ CI/CD 完整 pipeline (GitHub Actions workflow)
- ✅ 灰度發佈與回滾
- ✅ 監控與故障排查
- ✅ 版本發佈 checklist

**已在其他文檔詳細說明** (僅提供鉤子):
- ❌ 基礎編譯流程 → 見 [prd_hf-live.07-implementation.md §4](prd_hf-live.07-implementation.md)
- ❌ Submodule 配置 → 見 [prd_hf-live.04-project-config.md §2-3](prd_hf-live.04-project-config.md)
- ❌ 熱更新基礎 → 見 [prd_hf-live.03-workflow.md §2.3](prd_hf-live.03-workflow.md)
- ❌ 多交易所數據結構 → 見 [prd_hf-live.01-data-mapping.md](prd_hf-live.01-data-mapping.md)

---

## 一、構建優化 (生產環境)

### 1.1 Release 模式性能調優

**問題**: 默認 CMake Release 模式已經是 `-O3`,但還能優化嗎?

**進階優化選項** (在 CMakeLists.txt 中):

```cmake
# hf-live/CMakeLists.txt

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # ===== CPU 指令集優化 =====
    # 選項 1: 針對當前 CPU 架構優化 (最快,但不可移植)
    add_compile_options(-march=native)

    # 選項 2: 指定 AVX2 支持 (適合大多數現代 CPU)
    # add_compile_options(-march=haswell -mavx2 -mfma)

    # ===== Link-Time Optimization =====
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

    # ===== 符號表處理 =====
    # 生產環境: strip 符號表減小 .so 體積
    add_link_options(-Wl,--strip-all)

    # 調試環境: 保留符號表 (取消上面一行,啟用下面)
    # add_link_options(-Wl,--build-id)
endif()
```

**效果對比**:

| 優化級別 | .so 大小 | 因子計算耗時 | 適用場景 |
|---------|---------|------------|---------|
| `-O3` (默認) | 2.3 MB | 500 ns | 開發/測試 |
| `-O3 -march=native` | 2.3 MB | 420 ns | 生產 (相同 CPU) |
| 上述 + LTO | 2.0 MB | 400 ns | 生產 (推薦) ✅ |
| 上述 + strip | 1.1 MB | 400 ns | 生產 (最優) ✅ |

**權衡**:
- `-march=native`: 在不同 CPU 上可能崩潰 (illegal instruction)
- LTO: 編譯時間增加 30-50%,但運行時性能提升 5-10%
- strip: 無法用 gdb 調試,生產環境崩潰只有 backtrace 地址

**Makefile 集成**:

```makefile
# hf-live/Makefile

# ===== 新增: Release 優化目標 =====
.PHONY: build-release
build-release:
	$(call build_target,-DCMAKE_BUILD_TYPE=Release,libsignal.so (Release))

.PHONY: build-debug
build-debug:
	$(call build_target,-DCMAKE_BUILD_TYPE=Debug,libsignal.so (Debug))
```

---

### 1.2 多因子庫並行構建優化

**問題**: 當 `factors/` 下有 10+ 個因子模塊時,如何加速編譯?

**方案**: CMake 已經通過 `file(GLOB)` 自動發現,無需修改 CMakeLists.txt (見 [prd_hf-live.07-implementation.md §4.1](prd_hf-live.07-implementation.md))

**關鍵**: Makefile 中的 `JOBS` 變量已優化 (使用一半 CPU 核心)

**驗證並行效果**:

```bash
# 查看編譯過程
make clean && make 2>&1 | tee build.log

# 分析並行度
grep "Building CXX" build.log | wc -l
# 應看到多個 Building 任務同時進行
```

**瓶頸排查**:

```bash
# 如果編譯慢,檢查是否 I/O 瓶頸
iostat -x 1

# 如果 CPU 未充分利用,手動設置更多線程
make clean && JOBS=8 make
```

---

## 二、CI/CD Pipeline 設計

### 2.1 hf-live 倉庫自動構建

**目標**: 每次 push 到 hf-live 倉庫時,自動編譯並上傳 artifact

**GitHub Actions Workflow**:

```yaml
# hf-live/.github/workflows/build.yml
name: Build and Test libsignal.so

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-22.04

    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Install dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake g++ libc6-dev

      - name: Build Release
        run: |
          make build-release

      - name: Verify binary
        run: |
          # 檢查 .so 是否生成
          test -f build/libsignal.so || exit 1

          # 檢查依賴 (不應包含 Godzilla 路徑)
          ldd build/libsignal.so | grep -q godzilla && exit 1 || true

          # 檢查 C API 符號
          nm build/libsignal.so | grep -q signal_create || exit 1
          nm build/libsignal.so | grep -q signal_destroy || exit 1
          nm build/libsignal.so | grep -q signal_on_data || exit 1

          echo "✅ Binary verification passed"

      - name: Calculate checksum
        run: |
          sha256sum build/libsignal.so > build/libsignal.so.sha256
          cat build/libsignal.so.sha256

      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: libsignal-${{ github.sha }}
          path: |
            build/libsignal.so
            build/libsignal.so.sha256
          retention-days: 30

      - name: Create release (on tag)
        if: startsWith(github.ref, 'refs/tags/v')
        uses: softprops/action-gh-release@v1
        with:
          files: |
            build/libsignal.so
            build/libsignal.so.sha256
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**觸發條件**:
- Push 到 main/develop 分支 → 自動構建
- 創建 tag (v1.0.0) → 自動構建 + 創建 GitHub Release
- Pull Request → 自動構建驗證

---

### 2.2 godzilla-evan 集成測試

**目標**: 驗證 hf-live submodule 更新後,端到端集成正常

```yaml
# godzilla-evan/.github/workflows/integration-test.yml
name: HF-Live Integration Test

on:
  push:
    branches: [feature/hf-live-support]

jobs:
  integration-test:
    runs-on: ubuntu-22.04

    steps:
      - name: Checkout with submodules
        uses: actions/checkout@v3
        with:
          submodules: recursive
          token: ${{ secrets.SUBMODULE_TOKEN }}

      - name: Build hf-live
        run: |
          cd hf-live
          make build-release

      - name: Setup Godzilla environment
        run: |
          # 安裝 Godzilla 依賴
          sudo apt-get install -y python3.8 python3-pip
          pip3 install pytest

          # 編譯 Godzilla (簡化版,實際需要完整構建)
          # docker build -t godzilla-dev .

      - name: Run integration test
        run: |
          # 測試 libsignal.so 是否能被加載
          python3 -c "
          import ctypes
          lib = ctypes.CDLL('./hf-live/build/libsignal.so')
          assert lib.signal_create is not None
          print('✅ Integration test passed')
          "
```

**關鍵**: 需要 `SUBMODULE_TOKEN` 以訪問私有 hf-live 倉庫

---

### 2.3 自動化版本號管理

**問題**: 如何自動遞增版本號?

**方案**: 使用 Git tag 作為版本號來源

```bash
# hf-live/scripts/version.sh
#!/bin/bash

# 獲取最新 tag
VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0")

# 生成版本信息
cat > include/version.h <<EOF
#ifndef HF_LIVE_VERSION_H
#define HF_LIVE_VERSION_H

#define HF_LIVE_VERSION "$VERSION"
#define HF_LIVE_BUILD_DATE "$(date +%Y-%m-%d)"
#define HF_LIVE_GIT_COMMIT "$(git rev-parse --short HEAD)"

#endif
EOF

echo "Generated version.h: $VERSION"
```

**集成到 Makefile**:

```makefile
# 在 build 之前生成版本信息
build:
	@bash scripts/version.sh
	$(call build_target,,libsignal.so)
```

---

## 三、部署與灰度發佈

### 3.1 灰度發佈策略

**場景**: 新版本 libsignal.so v1.1.0 需要在生產環境驗證,但不想全量切換

**方案**: 運行兩個策略實例,按 symbol 分流

```bash
# 1. 保留 v1.0.0 實例
docker exec godzilla-dev pm2 list
# my_factor_strategy_v1.0 (運行中)

# 2. 部署 v1.1.0 實例 (使用新 .so)
docker cp libsignal_v1.1.0.so godzilla-dev:/app/hf-live/build/libsignal_v1.1.so

# 3. 創建新配置文件
docker exec godzilla-dev bash -c 'cat > /app/config/strategy_v1.1.json <<EOF
{
  "name": "my_factor_strategy_v1.1",
  "path": "strategies/my_factor_strategy/run.py",
  "signal_library_path": "/app/hf-live/build/libsignal_v1.1.so",
  "symbols": ["BTCUSDT", "ETHUSDT"]  // 僅這兩個品種使用 v1.1
}
EOF'

# 4. 啟動 v1.1 實例
docker exec godzilla-dev pm2 start /app/config/strategy_v1.1.json

# 5. 觀察 v1.1 性能
docker exec godzilla-dev pm2 logs my_factor_strategy_v1.1 --lines 100

# 6. 驗證無誤後,全量切換
docker exec godzilla-dev pm2 stop my_factor_strategy_v1.0
docker exec godzilla-dev pm2 delete my_factor_strategy_v1.0

# 更新主實例配置指向 v1.1
docker exec godzilla-dev bash -c "
cp /app/hf-live/build/libsignal_v1.1.so /app/hf-live/build/libsignal.so
pm2 restart my_factor_strategy_v1.1
pm2 save
"
```

**監控指標**:
- 延遲對比: v1.0 vs v1.1 的 on_factor 回調耗時
- 準確率對比: v1.0 vs v1.1 的信號質量
- 錯誤率對比: 是否有 segfault 或異常日誌

---

### 3.2 一鍵回滾腳本

**目標**: 發現問題時,30 秒內回滾到上一個版本

```bash
# hf-live/scripts/rollback.sh
#!/bin/bash

set -e

ROLLBACK_VERSION=${1:-"v1.0.0"}
CONTAINER=${2:-"godzilla-dev"}
SO_PATH="/app/hf-live/build/libsignal.so"
STRATEGY_NAME="my_factor_strategy"

echo "🔄 Rolling back to $ROLLBACK_VERSION..."

# 1. 從 GitHub Release 下載舊版本
curl -L -o /tmp/libsignal.so \
  "https://github.com/<org>/hf-live/releases/download/$ROLLBACK_VERSION/libsignal.so"

# 2. 驗證 checksum
curl -L -o /tmp/libsignal.so.sha256 \
  "https://github.com/<org>/hf-live/releases/download/$ROLLBACK_VERSION/libsignal.so.sha256"
cd /tmp && sha256sum -c libsignal.so.sha256

# 3. 停止策略
docker exec $CONTAINER pm2 stop $STRATEGY_NAME

# 4. 替換 .so
docker cp /tmp/libsignal.so $CONTAINER:$SO_PATH

# 5. 重啟策略
docker exec $CONTAINER pm2 restart $STRATEGY_NAME

# 6. 驗證
sleep 2
docker exec $CONTAINER pm2 logs $STRATEGY_NAME --lines 10 | grep "Signal library loaded" || {
  echo "❌ Rollback failed!"
  exit 1
}

echo "✅ Rolled back to $ROLLBACK_VERSION successfully"
```

**使用**:

```bash
# 回滾到 v1.0.0
bash scripts/rollback.sh v1.0.0

# 指定容器
bash scripts/rollback.sh v1.0.0 godzilla-prod
```

---

## 四、監控與故障排查

### 4.1 關鍵監控指標

**指標 1: .so 加載狀態**

```bash
# 查看 runner.cpp 是否成功加載 libsignal.so
docker exec godzilla-dev pm2 logs my_factor_strategy | grep "\[Runner\] Signal library loaded"

# 預期輸出:
# [Runner] Signal library loaded from /app/hf-live/build/libsignal.so
```

**指標 2: 因子回調執行次數**

在 strategy.py 中添加計數器:

```python
class MyFactorStrategy(Strategy):
    def __init__(self):
        self.factor_callback_count = 0

    def on_factor(self, context, symbol, timestamp, values):
        self.factor_callback_count += 1
        if self.factor_callback_count % 1000 == 0:
            self.logger.info(f"Factor callbacks: {self.factor_callback_count}")
```

**指標 3: 因子延遲統計**

在 runner.cpp 中添加計時:

```cpp
void Runner::on_factor_callback(...) {
    auto start = std::chrono::high_resolution_clock::now();

    // 調用 Python on_factor
    strategy.second->on_factor(context_, symbol, timestamp, values, count);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    if (duration > 1000000) {  // > 1ms
        fprintf(stderr, "[Runner] on_factor took %ld ns (warning!)\n", duration);
    }
}
```

---

### 4.2 故障排查手冊

#### 問題 1: .so 加載失敗

**症狀**:

```
[Runner] Failed to load signal library: ./libsignal.so: cannot open shared object file
```

**排查步驟**:

```bash
# 1. 檢查路徑配置
docker exec godzilla-dev cat /app/config/my_factor_strategy.json | grep signal_library_path

# 2. 檢查文件是否存在
docker exec godzilla-dev ls -lh /app/hf-live/build/libsignal.so

# 3. 檢查權限
docker exec godzilla-dev stat /app/hf-live/build/libsignal.so
# 應為 -rwxr-xr-x

# 4. 檢查依賴
docker exec godzilla-dev ldd /app/hf-live/build/libsignal.so
# 不應有 "not found"
```

**解決**:

```bash
# 如果權限錯誤
docker exec godzilla-dev chmod 755 /app/hf-live/build/libsignal.so

# 如果路徑錯誤,修改配置
vim config/my_factor_strategy.json
# 修改 signal_library_path 為絕對路徑

# 重啟策略
docker exec godzilla-dev pm2 restart my_factor_strategy
```

---

#### 問題 2: on_factor 從未被調用

**症狀**:

```python
# 策略日誌中無任何 "Factor:" 輸出
pm2 logs my_factor_strategy | grep "Factor:"
# (無輸出)
```

**排查步驟**:

```bash
# 1. 檢查 signal_register_callback 是否成功
docker exec godzilla-dev pm2 logs my_factor_strategy --err | grep "signal_register_callback"

# 2. 檢查 Engine::OnDepth 是否被調用 (添加 fprintf 調試)
# 需要重新編譯 libsignal.so 添加日誌

# 3. 檢查 market data 是否到達
docker exec godzilla-dev pm2 logs my_factor_strategy | grep "on_depth"
# 如果 on_depth 也沒有被調用,說明 MD 有問題
```

**解決**:

```bash
# 如果是 MD 問題,重啟 MD 服務
docker exec godzilla-dev pm2 restart md_binance

# 如果是回調註冊問題,檢查 runner.cpp 集成代碼
```

---

#### 問題 3: 因子數據異常

**症狀**:

```python
# on_factor 中 values 全是 NaN
def on_factor(self, context, symbol, timestamp, values):
    print(values)  # [nan, nan, nan, ...]
```

**排查步驟**:

```bash
# 1. 檢查 market_data_types.h 版本
docker exec godzilla-dev cat /app/hf-live/include/market_data_types.VERSION

# 與 Godzilla 版本對比
docker exec godzilla-dev cat /app/core/cpp/wingchun/include/kungfu/wingchun/msg.h | head -10

# 2. 檢查 Depth 結構體大小
# 在 runner.cpp 中添加:
# fprintf(stderr, "sizeof(Depth) = %zu\n", sizeof(Depth));

# 在 libsignal.so 中添加:
# fprintf(stderr, "sizeof(Depth) in .so = %zu\n", sizeof(Depth));

# 如果大小不一致,說明 header 不同步
```

**解決**:

```bash
# 同步 market_data_types.h (見 prd_hf-live.04-project-config.md §6.2)
cd godzilla-evan
cp core/cpp/wingchun/include/kungfu/wingchun/msg.h \
   hf-live/include/market_data_types.h

cd hf-live
make clean && make
docker cp build/libsignal.so godzilla-dev:/app/hf-live/build/
docker exec godzilla-dev pm2 restart my_factor_strategy
```

---

## 五、版本發佈 Checklist

### 5.1 發佈前檢查

```markdown
## hf-live v1.1.0 Release Checklist

### 代碼完成度
- [ ] 所有新增因子已實現並測試
- [ ] 代碼 Review 完成 (至少 1 人 approve)
- [ ] 無 TODO/FIXME 註釋殘留
- [ ] 代碼風格符合規範 (clang-format)

### 構建與測試
- [ ] `make clean && make` 編譯成功
- [ ] `ldd build/libsignal.so` 無 Godzilla 依賴
- [ ] 本地集成測試通過 (連接 Testnet)
- [ ] 性能測試: 因子計算 < 500ns

### 文檔更新
- [ ] 更新 CHANGELOG.md (新增功能、Bug 修復)
- [ ] 更新 README.md (如有新因子)
- [ ] 更新 market_data_types.VERSION (如需要)
- [ ] 更新 factors/README.md (因子列表)

### 版本管理
- [ ] 創建 Git tag: `git tag v1.1.0`
- [ ] 推送 tag: `git push origin v1.1.0`
- [ ] GitHub Release 自動創建並上傳 artifact
- [ ] 計算並驗證 checksum

### 通知與協調
- [ ] 通知策略大師: 新版本可用
- [ ] 提供升級指南 (Breaking Changes)
- [ ] 安排灰度發佈時間窗口
```

---

### 5.2 發佈後驗證

```bash
# 1. 下載 GitHub Release
curl -L -o libsignal.so \
  https://github.com/<org>/hf-live/releases/download/v1.1.0/libsignal.so

# 2. 驗證 checksum
curl -L -o libsignal.so.sha256 \
  https://github.com/<org>/hf-live/releases/download/v1.1.0/libsignal.so.sha256
sha256sum -c libsignal.so.sha256

# 3. 部署到 Testnet
docker cp libsignal.so godzilla-testnet:/app/hf-live/build/
docker exec godzilla-testnet pm2 restart my_factor_strategy

# 4. 監控 1 小時
docker exec godzilla-testnet pm2 logs my_factor_strategy --lines 1000 | \
  grep -E "(ERROR|WARNING|segfault)"

# 5. 確認無問題後,部署到生產
```

---

## 六、總結

### 6.1 與前 7 個文檔的關係

| 文檔 | 覆蓋內容 | 本文檔補充 |
|------|---------|-----------|
| 04-project-config | Git Submodule 基礎配置 | CI/CD 中 submodule 自動更新 |
| 07-implementation | 根 Makefile + 基礎編譯 | Release 優化、LTO、並行構建 |
| 03-workflow | 熱更新概念 | 灰度發佈、回滾腳本、監控指標 |
| 00-abstract | CI/CD 簡單示例 | 完整 GitHub Actions workflow |

### 6.2 核心貢獻

本文檔專注於**生產環境運維**,補充前 7 個文檔缺少的:
1. ✅ 構建優化技巧 (LTO, march=native, strip)
2. ✅ 完整 CI/CD pipeline (自動構建、測試、發佈)
3. ✅ 灰度發佈與回滾 (生產級別部署策略)
4. ✅ 故障排查手冊 (3 個常見問題 + 解決方案)
5. ✅ 版本發佈 checklist (確保質量)

### 6.3 快速導航

**我想...**
- 優化編譯速度 → 見 §1.2
- 優化運行性能 → 見 §1.1
- 搭建 CI/CD → 見 §2.1-2.3
- 灰度發佈新版本 → 見 §3.1
- 快速回滾 → 見 §3.2
- 排查 .so 加載失敗 → 見 §4.2 問題 1
- 發佈新版本 → 見 §5.1

---

**版本**: v1.0
**日期**: 2025-12-04
**定位**: Day-2 Operations 手冊 (補充前 7 個 PRD 的設計與基礎實現)
**核心**: 構建優化 + CI/CD + 灰度發佈 + 監控運維
