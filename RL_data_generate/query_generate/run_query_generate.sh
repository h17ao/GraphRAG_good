#!/bin/bash

# 激活conda环境并运行查询生成脚本
echo "正在激活digimon环境并运行查询生成脚本..."

# 初始化conda（使用conda.sh而不是bashrc）
eval "$(conda shell.bash hook)"

# 激活conda环境并运行脚本
conda activate digimon && PYTHONPATH=../.. python query_generate.py

echo "查询生成脚本执行完成" 