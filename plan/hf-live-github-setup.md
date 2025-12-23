# hf-live GitHub 倉庫設置指南

**日期**: 2025-12-23
**目標**: 將 hf-live 設置為獨立的 GitHub private 倉庫並配置為 godzilla-evan 的 submodule

---

## 當前狀態

**hf-live 本地倉庫狀態**:
- 目錄: `/home/huyifan/projects/godzilla-evan/hf-live`
- 當前分支: `feature/latency-monitoring` (HEAD: f0f2ef1)
- main 分支: 9de33f2 (較舊)
- 無遠端配置 (純本地倉庫)

**godzilla-evan 主倉庫記錄**:
- hf-live commit: f0f2ef1 (在 feature/latency-monitoring 分支)
- main 分支已推送到 GitHub

---

## Step 1: 在 GitHub 創建私有倉庫

### 1.1 手動在 GitHub 創建

前往: https://github.com/new

設置:
- **Repository name**: `hf-live`
- **Description**: "High-frequency live trading signal engine (private)"
- **Visibility**: ✅ **Private**
- **Initialize**: ⚠️ **不要**勾選任何選項 (保持空倉庫)

點擊 "Create repository"

### 1.2 記錄倉庫 URL

創建後,GitHub 會顯示倉庫 URL:
```
https://github.com/eiahb3838ya/hf-live.git
```

---

## Step 2: 整理 hf-live 分支

### 2.1 清理本地編譯產物

```bash
cd /home/huyifan/projects/godzilla-evan/hf-live

# 刪除編譯產物 (需要在 container 內執行)
docker exec godzilla-dev bash -c "cd /app/hf-live && rm -rf build/ build_debug/ || true"

# 或在 host 上強制刪除
sudo rm -rf build/ build_debug/
```

### 2.2 合併 feature 分支到 main

```bash
# 切換到 main 分支
git checkout main

# 合併 feature/latency-monitoring (包含所有 Phase 6 修復)
git merge feature/latency-monitoring --no-ff -m "merge: integrate Phase 6 features and fixes into main

## Phase 6 Features (feature/latency-monitoring → main)

### Critical Fixes
- f0f2ef1: fix(factor): Factor 12 ticker_momentum semantic correctness
- 07bcbbf: fix(callback): symbol normalization and model output queue
- 31fdeef: fix(model): change model from test0000 to linear

### Production Readiness
- 11c8791: build: recompile with DEBUG_MODE=OFF
- b9d6b79: build: add DEBUG_MODE support
- 8abe534: feat(debug): add DEBUG_MODE option for observability
- badf70b: perf(logging): remove per-tick verbose logs

### Full Market Data Integration
- 8100093: feat(phase-6): implement full market data with 15 factors
- c92bb6b: feat(phase-5d): zero-interface latency monitoring

This merge brings hf-live main branch up to date with all Phase 6
production-ready features and critical bug fixes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Step 3: 添加 GitHub 遠端並推送

### 3.1 添加遠端

```bash
# 在 hf-live 目錄
cd /home/huyifan/projects/godzilla-evan/hf-live

# 添加 GitHub 遠端 (使用你的用戶名)
git remote add origin https://github.com/eiahb3838ya/hf-live.git

# 驗證遠端配置
git remote -v
# 應該看到:
# origin	https://github.com/eiahb3838ya/hf-live.git (fetch)
# origin	https://github.com/eiahb3838ya/hf-live.git (push)
```

### 3.2 推送 main 分支

```bash
# 推送 main 分支 (設置為默認 upstream)
git push -u origin main

# 推送其他分支 (可選)
git push origin feature/latency-monitoring
```

---

## Step 4: 配置 godzilla-evan 使用 submodule

### 4.1 移除舊的 hf-live 目錄

⚠️ **重要**: 先備份本地修改!

```bash
cd /home/huyifan/projects/godzilla-evan

# 檢查是否有未提交的修改
cd hf-live && git status && cd ..

# 移除 hf-live 目錄 (主倉庫會顯示為刪除)
rm -rf hf-live
```

### 4.2 添加為 Git Submodule

```bash
cd /home/huyifan/projects/godzilla-evan

# 添加 submodule (指向正確的 commit)
git submodule add https://github.com/eiahb3838ya/hf-live.git hf-live

# 切換到正確的 commit (f0f2ef1)
cd hf-live
git checkout f0f2ef1
cd ..

# 暫存 submodule 配置
git add .gitmodules hf-live
```

### 4.3 提交 submodule 配置

```bash
git commit -m "config: convert hf-live to GitHub submodule

- Add hf-live as Git submodule pointing to private GitHub repo
- Pin to commit f0f2ef1 (Phase 6 production-ready state)
- This allows independent version control for hf-live

Submodule URL: https://github.com/eiahb3838ya/hf-live.git

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 4.4 推送到遠端

```bash
git push origin main
```

---

## Step 5: 驗證配置

### 5.1 檢查 .gitmodules 文件

```bash
cat .gitmodules
```

應該看到:
```ini
[submodule "hf-live"]
	path = hf-live
	url = https://github.com/eiahb3838ya/hf-live.git
```

### 5.2 驗證 submodule 狀態

```bash
git submodule status
# 應該看到: f0f2ef1e0d56385d1df562b4bb3caf39b55e65a7 hf-live (heads/main)
```

### 5.3 測試 submodule 克隆

在另一個目錄測試:
```bash
cd /tmp
git clone https://github.com/eiahb3838ya/godzilla-evan.git test-clone
cd test-clone

# 初始化並更新 submodule
git submodule init
git submodule update

# 驗證 hf-live 目錄存在且指向正確 commit
cd hf-live
git log --oneline -1
# 應該看到: f0f2ef1 fix(factor): 修正 Factor 12 ticker_momentum 計算邏輯
```

---

## Step 6: 更新 Docker 容器配置

### 6.1 重新構建容器

如果 Dockerfile 有依賴 hf-live 的編譯步驟:

```bash
cd /home/huyifan/projects/godzilla-evan

# 確保 submodule 已更新
git submodule update --init --recursive

# 重新構建
docker-compose build godzilla-dev
```

### 6.2 容器內編譯 hf-live

```bash
docker exec godzilla-dev bash -c "cd /app/hf-live && mkdir -p build && cd build && cmake .. && make -j$(nproc)"
```

---

## 未來工作流程

### 開發新功能

```bash
# 在 hf-live 倉庫
cd /home/huyifan/projects/godzilla-evan/hf-live

# 創建新分支
git checkout -b feature/new-feature

# 開發並提交
git add .
git commit -m "feat: add new feature"

# 推送到 GitHub
git push origin feature/new-feature
```

### 在主倉庫更新 submodule 指向

```bash
# 在主倉庫
cd /home/huyifan/projects/godzilla-evan

# 進入 submodule 並切換到新 commit
cd hf-live
git fetch origin
git checkout <new-commit-hash>
cd ..

# 提交 submodule 更新
git add hf-live
git commit -m "chore: update hf-live to <commit-hash>

<描述更新內容>"
git push origin main
```

### 團隊成員克隆項目

```bash
# 克隆主倉庫並自動初始化 submodule
git clone --recursive https://github.com/eiahb3838ya/godzilla-evan.git

# 或分步驟:
git clone https://github.com/eiahb3838ya/godzilla-evan.git
cd godzilla-evan
git submodule init
git submodule update
```

---

## 注意事項

### 1. 私有倉庫權限

hf-live 是 private 倉庫,團隊成員需要:
- 被添加為倉庫 Collaborator
- 或使用 SSH key 進行認證

### 2. Submodule Commit Pin

主倉庫會記錄 submodule 的具體 commit hash:
- ⚠️ submodule 不會自動跟蹤分支
- ✅ 每次更新需要手動切換並提交

### 3. 編譯產物管理

`.gitignore` 應該包含:
```gitignore
# hf-live
build/
build_debug/
*.so
*.a
```

### 4. CI/CD 配置

如果有 CI/CD pipeline,需要:
- 配置 GitHub token 以訪問私有 submodule
- 在 CI 腳本中添加 `git submodule update --init --recursive`

---

## 回滾方案

如果設置過程出現問題:

### 回滾 submodule 配置

```bash
cd /home/huyifan/projects/godzilla-evan

# 移除 submodule
git submodule deinit -f hf-live
git rm -f hf-live
rm -rf .git/modules/hf-live

# 恢復原來的 hf-live 目錄
git checkout HEAD -- hf-live
```

### 保留本地備份

在執行任何操作前:
```bash
cp -r /home/huyifan/projects/godzilla-evan/hf-live /home/huyifan/hf-live-backup-20251223
```

---

## 參考資料

- [Git Submodules 官方文檔](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [GitHub Private Repos](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories#about-repository-visibility)
- 主倉庫: https://github.com/eiahb3838ya/godzilla-evan
- hf-live 倉庫: https://github.com/eiahb3838ya/hf-live (待創建)

---

**創建日期**: 2025-12-23
**狀態**: 等待執行
**優先級**: 中 (非緊急,但建議盡快完成以便版本管理)
