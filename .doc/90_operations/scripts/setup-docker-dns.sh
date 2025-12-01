#!/bin/bash

# 创建 Docker daemon 配置文件以解决 WSL2 DNS 问题

echo "正在配置 Docker daemon DNS 设置..."

# 创建 /etc/docker 目录（如果不存在）
sudo mkdir -p /etc/docker

# 创建 daemon.json 配置文件
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "dns": ["8.8.8.8", "114.114.114.114", "1.1.1.1"],
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://registry.docker-cn.com"
  ]
}
EOF

echo "配置文件已创建："
sudo cat /etc/docker/daemon.json

echo ""
echo "重启 Docker 服务..."
sudo service docker restart

echo ""
echo "等待 Docker 服务启动..."
sleep 3

echo "Docker 服务状态："
sudo service docker status

echo ""
echo "DNS 配置完成！现在可以运行 docker-compose up -d --build"

