#!/bin/bash
# WSL 中每日自动 pull 脚本
# 使用方法：将此脚本添加到 crontab 或在 Windows 任务计划中调用

REPO_DIR="/home/huyifan/projects/godzilla-evan/ref/hf-open-live-demo"
REPO_URL="http://172.16.12.71/gaowang/hf-open-live-demo.git"
LOG_FILE="/home/huyifan/projects/godzilla-evan/ref/pull_log.txt"

echo "========================================" | tee -a "$LOG_FILE"
echo "Daily Pull - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# 检查仓库是否存在
if [ ! -d "$REPO_DIR" ]; then
    echo "Repository not found, cloning..." | tee -a "$LOG_FILE"
    git clone "$REPO_URL" "$REPO_DIR" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully cloned repository" | tee -a "$LOG_FILE"
    else
        echo "✗ Failed to clone repository" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "Repository exists, pulling updates..." | tee -a "$LOG_FILE"
    cd "$REPO_DIR"
    
    # 保存当前提交
    BEFORE_COMMIT=$(git rev-parse HEAD 2>/dev/null)
    
    # Pull 更新
    git fetch origin 2>&1 | tee -a "$LOG_FILE"
    git pull origin master 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        AFTER_COMMIT=$(git rev-parse HEAD)
        
        if [ "$BEFORE_COMMIT" != "$AFTER_COMMIT" ]; then
            echo "✓ Updates pulled successfully" | tee -a "$LOG_FILE"
            echo "Changed from $BEFORE_COMMIT to $AFTER_COMMIT" | tee -a "$LOG_FILE"
        else
            echo "✓ Already up to date" | tee -a "$LOG_FILE"
        fi
    else
        echo "✗ Failed to pull updates" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

echo "" | tee -a "$LOG_FILE"


