#!/bin/bash

# SFT数据生成脚本
# 使用方法: ./run_sft_generate.sh <输入文件> <输出文件> [并发数]

# 检查参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <输入文件> <输出文件> [并发数]"
    echo "示例: $0 input_data.json sft_output.jsonl 30"
    exit 1
fi

INPUT_FILE=$1
OUTPUT_FILE=$2
MAX_CONCURRENT=${3:-50}  # 默认并发数为50

echo "🚀 开始SFT数据生成..."
echo "📁 输入文件: $INPUT_FILE"
echo "📁 输出文件: $OUTPUT_FILE"
echo "⚡ 最大并发数: $MAX_CONCURRENT"

# 保存原始目录和处理文件路径
SCRIPT_DIR="$(dirname "$0")"
ORIGINAL_DIR="$(pwd)"

# 如果输入文件是相对路径，转换为绝对路径
if [[ ! "$INPUT_FILE" = /* ]]; then
    INPUT_FILE="$ORIGINAL_DIR/$INPUT_FILE"
fi

# 如果输出文件是相对路径，转换为绝对路径  
if [[ ! "$OUTPUT_FILE" = /* ]]; then
    OUTPUT_FILE="$ORIGINAL_DIR/$OUTPUT_FILE"
fi

# 切换到GraphRAG_new根目录
cd "$SCRIPT_DIR/../.."

# 初始化conda（如果使用conda环境）
if command -v conda &> /dev/null; then
    echo "正在激活digimon环境..."
    eval "$(conda shell.bash hook)"
    conda activate digimon
fi

# 设置PYTHONPATH
export PYTHONPATH=.

# 运行Python脚本（使用绝对路径）
python RL_data_generate/sft_generate/sft_generate.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_FILE" \
    --max_concurrent "$MAX_CONCURRENT"

echo "✅ SFT数据生成完成！"
