#!/bin/bash

# GraphRAG环境自动设置脚本
# 检查digimon环境，如果不存在则创建，然后自动修复ColBERT问题，下载NLTK数据，设置NLTK符号链接

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_progress() {
    echo -e "${BLUE}🔄 $1${NC}"
}

# 检查conda是否可用
check_conda() {
    if ! command -v conda &> /dev/null; then
        log_error "未找到conda命令，请先安装Anaconda或Miniconda"
        exit 1
    fi
    log_success "Conda已安装"
}

# 检查是否存在digimon环境
check_digimon_env() {
    log_info "检查digimon conda环境..."
    
    if conda env list | grep -q "^digimon\s"; then
        log_success "发现digimon环境"
        return 0
    else
        log_warning "未发现digimon环境"
        return 1
    fi
}

# 创建digimon环境
create_digimon_env() {
    log_progress "开始创建digimon环境..."
    
    if [ ! -f "digimon_environment.yml" ]; then
        log_error "未找到digimon_environment.yml文件"
        exit 1
    fi
    
    log_info "使用digimon_environment.yml创建环境（这可能需要几分钟）..."
    conda env create -f digimon_environment.yml -n digimon
    
    log_success "digimon环境创建完成"
}

# 修复ColBERT AdamW导入问题
fix_colbert() {
    log_progress "修复ColBERT AdamW导入问题..."
    
    # 激活环境并检查ColBERT
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate digimon
    
    # 找到colbert包路径
    COLBERT_PATH=$(python -c "import site; import os; print([p for p in site.getsitepackages() if os.path.exists(os.path.join(p, 'colbert'))][0])" 2>/dev/null)
    
    if [ -z "$COLBERT_PATH" ]; then
        log_error "未找到ColBERT包路径"
        return 1
    fi
    
    TRAINING_FILE="$COLBERT_PATH/colbert/training/training.py"
    
    if [ ! -f "$TRAINING_FILE" ]; then
        log_error "未找到training.py文件: $TRAINING_FILE"
        return 1
    fi
    
    # 检查是否需要修复
    if grep -q "from transformers import AdamW," "$TRAINING_FILE"; then
        log_info "发现需要修复的AdamW导入，正在修复..."
        
        # 备份原文件
        cp "$TRAINING_FILE" "$TRAINING_FILE.backup"
        
        # 执行替换
        sed -i 's/from transformers import AdamW, get_linear_schedule_with_warmup/from torch.optim import AdamW\nfrom transformers import get_linear_schedule_with_warmup/' "$TRAINING_FILE"
        
        log_success "ColBERT AdamW导入问题已修复"
    else
        log_success "ColBERT无需修复或已修复"
    fi
    
    # 验证修复
    if python -c "import colbert" 2>/dev/null; then
        log_success "ColBERT导入测试通过"
    else
        log_error "ColBERT导入测试失败"
        return 1
    fi
}

# 检查并下载NLTK数据
check_and_download_nltk_data() {
    log_progress "检查NLTK数据..."
    
    # 确保激活了digimon环境
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate digimon
    
    # 我们的cache目录路径（相对于项目根目录）
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CACHE_NLTK_DATA="$(dirname "$SCRIPT_DIR")/cache/nltk_data"
    
    # 确保cache目录存在
    mkdir -p "$CACHE_NLTK_DATA"
    
    # 检查NLTK数据是否存在
    python -c "
import nltk
import sys
import os

nltk_path = '$CACHE_NLTK_DATA'
if nltk_path not in nltk.data.path:
    nltk.data.path.insert(0, nltk_path)

required = {'punkt': 'tokenizers/punkt', 'wordnet': 'corpora/wordnet'}
missing = []

for name, path in required.items():
    try:
        nltk.data.find(path)
        print(f'✅ {name} 已存在')
    except LookupError:
        missing.append(name)
        print(f'❌ {name} 缺失')

if missing:
    print(f'📥 开始下载缺失的NLTK数据: {missing}')
    for package in missing:
        try:
            nltk.download(package, download_dir=nltk_path, quiet=False)
            print(f'✅ {package} 下载完成')
        except Exception as e:
            print(f'❌ {package} 下载失败: {e}')
            sys.exit(1)
    print('🎉 所有NLTK数据已准备就绪')
else:
    print('✅ 所有NLTK数据已存在')
"
    
    if [ $? -eq 0 ]; then
        log_success "NLTK数据检查/下载完成"
        return 0
    else
        log_error "NLTK数据检查/下载失败"
        return 1
    fi
}

# 设置NLTK符号链接
setup_nltk_symlink() {
    log_progress "设置NLTK符号链接..."
    
    # 确保激活了digimon环境
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate digimon
    
    # 找到llama_index的nltk_cache路径
    SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
    LLAMA_NLTK_CACHE="$SITE_PACKAGES/llama_index/legacy/_static/nltk_cache"
    
    # 我们的cache目录路径（相对于项目根目录）
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    CACHE_NLTK_DATA="$(dirname "$SCRIPT_DIR")/cache/nltk_data"
    
    if [ ! -d "$CACHE_NLTK_DATA" ]; then
        log_error "缓存NLTK数据不存在: $CACHE_NLTK_DATA"
        log_info "请确保已正确设置cache目录"
        return 1
    fi
    
    log_info "NLTK缓存路径: $LLAMA_NLTK_CACHE"
    log_info "目标数据路径: $CACHE_NLTK_DATA"
    
    # 检查是否已经是正确的符号链接
    if [ -L "$LLAMA_NLTK_CACHE" ] && [ "$(readlink -f "$LLAMA_NLTK_CACHE")" = "$(readlink -f "$CACHE_NLTK_DATA")" ]; then
        log_success "NLTK符号链接已正确设置"
        return 0
    fi
    
    # 删除现有目录或链接
    if [ -e "$LLAMA_NLTK_CACHE" ]; then
        log_info "删除现有NLTK缓存目录/链接..."
        rm -rf "$LLAMA_NLTK_CACHE"
    fi
    
    # 创建符号链接
    ln -s "$CACHE_NLTK_DATA" "$LLAMA_NLTK_CACHE"
    log_success "NLTK符号链接已创建: $LLAMA_NLTK_CACHE -> $CACHE_NLTK_DATA"
    
    # 验证符号链接
    if [ -L "$LLAMA_NLTK_CACHE" ] && [ -d "$LLAMA_NLTK_CACHE" ]; then
        log_success "NLTK符号链接验证通过"
    else
        log_error "NLTK符号链接验证失败"
        return 1
    fi
}

# 最终测试
final_test() {
    log_progress "执行最终测试..."
    
    # 激活环境
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate digimon
    
    # 测试所有关键组件
    python -c "
import sys
print('🔍 最终环境测试...')

try:
    import colbert
    print('✅ ColBERT: 正常')
except Exception as e:
    print(f'❌ ColBERT: {e}')
    sys.exit(1)

try:
    from Core.Utils.Evaluation import bleu_1
    print('✅ NLTK评估模块: 正常') 
except Exception as e:
    print(f'❌ NLTK评估模块: {e}')
    sys.exit(1)

try:
    import torch, transformers, sentence_transformers
    print('✅ AI核心包: 正常')
except Exception as e:
    print(f'❌ AI核心包: {e}')
    sys.exit(1)

print('🎉 所有测试通过！环境已准备就绪！')
"
    
    if [ $? -eq 0 ]; then
        log_success "最终测试通过"
        return 0
    else
        log_error "最终测试失败"
        return 1
    fi
}

# 显示使用方法
show_usage() {
    echo "GraphRAG环境自动设置完成！"
    echo ""
    echo "使用方法:"
    echo "  conda activate digimon"
    echo "  python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name hotpotqa"
    echo ""
    echo "如果遇到问题，可以手动运行:"
    echo "  python Core/Utils/EnvironmentSetup.py"
}

# 主函数
main() {
    echo "🚀 GraphRAG环境自动设置脚本"
    echo "================================"
    
    # 检查conda
    check_conda
    
    # 检查或创建digimon环境
    if check_digimon_env; then
        log_info "使用现有的digimon环境"
    else
        log_info "需要创建digimon环境"
        create_digimon_env
    fi
    
    # 修复ColBERT
    if ! fix_colbert; then
        log_error "ColBERT修复失败"
        exit 1
    fi
    
    # 检查并下载NLTK数据
    if ! check_and_download_nltk_data; then
        log_error "NLTK数据检查/下载失败"
        exit 1
    fi
    
    # 设置NLTK符号链接
    if ! setup_nltk_symlink; then
        log_error "NLTK符号链接设置失败"
        exit 1
    fi
    
    # 最终测试
    if ! final_test; then
        log_error "最终测试失败"
        exit 1
    fi
    
    echo ""
    log_success "🎉 环境设置完成！"
    echo ""
    show_usage
}

# 运行主函数
main "$@" 