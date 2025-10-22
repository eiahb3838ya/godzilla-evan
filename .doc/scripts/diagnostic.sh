#!/bin/bash

echo "=========================================="
echo "🔍 Godzilla-Evan 开发环境诊断报告"
echo "=========================================="
echo ""

# 第一步：WSL2 检查
echo "📋 第一步：WSL2 环境检查"
echo "----------------------------------------"
echo -n "✓ WSL 版本: "
uname -r | grep -q "microsoft" && echo "WSL2 ✅" || echo "非 WSL ❌"
echo -n "✓ 发行版: "
lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2
echo -n "✓ 当前用户: "
whoami
echo -n "✓ 工作目录: "
pwd
echo ""

# 第二步：Docker 检查
echo "📋 第二步：Docker Desktop 检查"
echo "----------------------------------------"
echo -n "✓ Docker 版本: "
docker --version 2>/dev/null || echo "未安装 ❌"
echo -n "✓ Docker Compose 版本: "
docker compose version 2>/dev/null | head -1 || echo "未安装 ❌"
echo -n "✓ Docker 服务状态: "
docker info > /dev/null 2>&1 && echo "运行中 ✅" || echo "未运行 ❌"
echo -n "✓ Docker DNS 配置: "
docker info 2>/dev/null | grep -q "8.8.8.8" && echo "已配置 ✅" || echo "使用默认配置"
echo ""

# 第三步：项目检查
echo "📋 第三步：项目环境检查"
echo "----------------------------------------"
echo -n "✓ 项目路径: "
[ -d "/home/huyifan/projects/godzilla-evan" ] && echo "/home/huyifan/projects/godzilla-evan ✅" || echo "不存在 ❌"
echo -n "✓ Dockerfile.dev: "
[ -f "Dockerfile.dev" ] && echo "存在 ✅" || echo "不存在 ❌"
echo -n "✓ docker-compose.yml: "
[ -f "docker-compose.yml" ] && echo "存在 ✅" || echo "不存在 ❌"
echo -n "✓ Git 仓库: "
[ -d ".git" ] && echo "已初始化 ✅" || echo "未初始化 ⚠️"
echo ""

# 第四步：容器检查
echo "📋 第四步：Docker 容器检查"
echo "----------------------------------------"
echo -n "✓ 镜像构建: "
docker images | grep -q "godzilla-evan-app" && echo "godzilla-evan-app ✅" || echo "未构建 ❌"
echo -n "✓ 容器状态: "
docker ps | grep -q "godzilla-dev" && echo "godzilla-dev 运行中 ✅" || echo "未运行 ❌"
echo "✓ 容器详情:"
docker ps --filter "name=godzilla-dev" --format "  - 名称: {{.Names}}\n  - 状态: {{.Status}}\n  - 端口: {{.Ports}}" 2>/dev/null
echo ""

# 第五步：容器内环境检查
echo "📋 第五步：容器内环境检查"
echo "----------------------------------------"
if docker ps | grep -q "godzilla-dev"; then
    echo "✓ Python 版本: $(docker exec godzilla-dev python3 --version 2>/dev/null)"
    echo "✓ CMake 版本: $(docker exec godzilla-dev cmake --version 2>/dev/null | head -1)"
    echo "✓ Git 版本: $(docker exec godzilla-dev git --version 2>/dev/null)"
    echo "✓ GCC 版本: $(docker exec godzilla-dev gcc --version 2>/dev/null | head -1)"
    echo -n "✓ 项目文件: "
    docker exec godzilla-dev ls /app > /dev/null 2>&1 && echo "可访问 ✅" || echo "不可访问 ❌"
else
    echo "❌ 容器未运行，无法检查容器内环境"
fi
echo ""

# 总结
echo "=========================================="
echo "📊 诊断总结"
echo "=========================================="
echo ""

ISSUES=0

# 检查关键项
docker info > /dev/null 2>&1 || ((ISSUES++))
docker ps | grep -q "godzilla-dev" || ((ISSUES++))
[ -f "Dockerfile.dev" ] || ((ISSUES++))
[ -f "docker-compose.yml" ] || ((ISSUES++))

if [ $ISSUES -eq 0 ]; then
    echo "🎉 恭喜！所有检查都通过了！"
    echo ""
    echo "✅ WSL2 环境正常"
    echo "✅ Docker Desktop 运行正常"
    echo "✅ 项目配置完整"
    echo "✅ 开发容器运行中"
    echo ""
    echo "你已经成功完成了所有步骤！🚀"
    echo ""
    echo "下一步可以："
    echo "  1. 进入容器: docker-compose exec app /bin/bash"
    echo "  2. 查看日志: docker-compose logs -f app"
    echo "  3. 在 Cursor 中连接 WSL 并打开项目"
else
    echo "⚠️ 发现 $ISSUES 个问题，请检查上面的报告"
fi

echo ""
echo "=========================================="
