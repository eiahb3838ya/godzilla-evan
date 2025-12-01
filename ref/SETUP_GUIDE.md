# 自动化 Git Pull 设置指南

## 问题背景

- WSL2 环境 + VPN 网络
- 需要访问内网 GitLab: http://172.16.12.71/gaowang/hf-open-live-demo
- 需要每天自动拉取更新

## 解决方案

已为您准备了两套方案：

---

## 方案 A：WSL 镜像网络模式（推荐）

### 1. 配置已创建

已在 `C:\Users\Administrator\.wslconfig` 创建配置文件，启用镜像网络模式。

### 2. 重启 WSL

在 Windows PowerShell 或 CMD 中执行：

```powershell
wsl --shutdown
```

等待 10 秒后重新打开 WSL。

### 3. 测试克隆

```bash
cd /home/huyifan/projects/godzilla-evan/ref
git clone http://172.16.12.71/gaowang/hf-open-live-demo.git
```

### 4. 设置自动化（使用 cron）

如果第 3 步成功，可以使用 WSL 的 cron：

```bash
# 编辑 crontab
crontab -e

# 添加以下行（每天早上 9 点执行）
0 9 * * * /home/huyifan/projects/godzilla-evan/ref/daily_pull.sh

# 或者每 6 小时执行一次
0 */6 * * * /home/huyifan/projects/godzilla-evan/ref/daily_pull.sh
```

查看日志：
```bash
cat /home/huyifan/projects/godzilla-evan/ref/pull_log.txt
```

---

## 方案 B：Windows 侧同步（备用方案）

如果方案 A 不工作，使用此方案。

### 1. 确保 Windows 安装了 Git

下载地址：https://git-scm.com/download/win

### 2. 首次运行脚本

双击桌面上的：`clone_and_sync.ps1`

或在 PowerShell 中执行：
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
C:\Users\Administrator\Desktop\clone_and_sync.ps1
```

### 3. 设置 Windows 任务计划程序

打开"任务计划程序" (Task Scheduler)：

1. 创建基本任务
2. 名称：`Daily Git Sync`
3. 触发器：每天 9:00 AM
4. 操作：启动程序
   - 程序：`powershell.exe`
   - 参数：`-ExecutionPolicy Bypass -File "C:\Users\Administrator\Desktop\clone_and_sync.ps1"`
5. 完成

#### 快速创建任务（PowerShell 命令）

```powershell
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-ExecutionPolicy Bypass -File "C:\Users\Administrator\Desktop\clone_and_sync.ps1"'
$trigger = New-ScheduledTaskTrigger -Daily -At 9am
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "DailyGitSync" -Action $action -Trigger $trigger -Settings $settings -Description "Daily sync of hf-open-live-demo repository"
```

---

## 文件说明

### WSL 文件

- `daily_pull.sh` - WSL 中的自动 pull 脚本
- `pull_log.txt` - 日志文件（运行后生成）
- `restart_wsl.txt` - WSL 重启说明

### Windows 文件

- `C:\Users\Administrator\.wslconfig` - WSL 配置文件
- `C:\Users\Administrator\Desktop\clone_and_sync.ps1` - Windows 同步脚本

---

## 测试脚本

### 测试 WSL 脚本

```bash
cd /home/huyifan/projects/godzilla-evan/ref
./daily_pull.sh
```

### 测试 Windows 脚本

```powershell
C:\Users\Administrator\Desktop\clone_and_sync.ps1
```

---

## 故障排除

### WSL 镜像网络模式不工作

症状：重启后仍无法克隆

解决：使用方案 B（Windows 侧同步）

### Windows Git 未找到

症状：脚本报错 "Git not found"

解决：
1. 安装 Git for Windows
2. 安装时选择"Add to PATH"
3. 重启 PowerShell

### WSL 路径访问失败

症状：无法访问 `\\wsl$\Ubuntu\...`

解决：
1. 确保 WSL 正在运行
2. 在文件资源管理器中访问 `\\wsl$` 查看可用的发行版名称
3. 修改脚本中的路径

---

## 下一步

1. **立即操作**：重启 WSL 测试方案 A
2. **如果失败**：运行方案 B 的脚本
3. **设置自动化**：根据成功的方案设置定时任务

有问题随时查看日志或重新运行脚本。


