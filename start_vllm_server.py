#!/usr/bin/env python3
"""
基于vLLM的本地LLM服务启动脚本 - Qwen3-8B推理（GPU 2&3，24GB*2显存，张量并行版）
"""

import subprocess
import sys
import os

# 模型配置
# MODEL_PATH = "/home/yh/cache/models/modelscope/hub/models/qwen/trained_qwen1.7b_step_200"
MODEL_PATH = "/home/yh/cache/models/modelscope/hub/models/qwen/Qwen3-4B"  # 第3个epoch训练好的模型路径
HOST = "0.0.0.0"
PORT = 8001

# ========== RTX 4090四卡分布式配置（张量并行）==========
GPU_MEMORY_UTILIZATION = 0.85  # 4090 24GB显存利用率 (70%，适应其他进程占用)
MAX_MODEL_LEN = 32768  # 扩展上下文长度，支持大规模文档处理 (32768 × 4)
MAX_NUM_SEQS = 100     # 4090并发数（适中配置）
MAX_NUM_BATCHED_TOKENS = 256  # 4090批处理大小（保守配置）
ENABLE_CHUNKED_PREFILL = True   # 分块预填充（减少峰值显存）
ENABLE_PREFIX_CACHING = False   # 暂时关闭前缀缓存（减少显存占用）
BLOCK_SIZE = 16                 # vLLM要求：块大小必须是16的倍数
# 注意：不使用量化参数，避免影响实验结果精度

# 设置使用的GPU设备 (使用四张4090: GPU 0,1,2,3)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 自动检测多卡
cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if cuda_visible_devices:
    device_list = [d for d in cuda_visible_devices.split(",") if d.strip() != ""]
    tensor_parallel_size = len(device_list)
else:
    tensor_parallel_size = 1

print("============================================")
print("🔥 GraphRAG vLLM本地服务启动器（RTX 4090双卡张量并行版 - Qwen3-1.7B）")
print("============================================")
print(f"[GPU检测] CUDA_VISIBLE_DEVICES = {cuda_visible_devices}")
print(f"[GPU检测] tensor_parallel_size = {tensor_parallel_size}")

# 检查模型是否存在
if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型路径不存在: {MODEL_PATH}")
    sys.exit(1)

# 检查vLLM
try:
    import vllm
    print("✅ vLLM已安装")
except ImportError:
    print("❌ vLLM未安装，请先安装vLLM！")
    sys.exit(1)

# 构建vLLM命令（全面性能优化，不影响精度）
cmd = [
    "python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_PATH,
    "--host", HOST,
    "--port", str(PORT),
    "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS),
    "--trust-remote-code",
    "--served-model-name", "qwen3-4b",
    "--block-size", str(BLOCK_SIZE),         # 显存块大小优化
    "--disable-custom-all-reduce",           # 单卡优化
    "--max-seq-len-to-capture", str(MAX_MODEL_LEN),  # 序列捕获优化
    # "--rope-scaling", '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}',  # RoPE缩放配置
]

# 添加显存优化选项
if ENABLE_CHUNKED_PREFILL:
    cmd.append("--enable-chunked-prefill")
if ENABLE_PREFIX_CACHING:
    cmd.append("--enable-prefix-caching")
if tensor_parallel_size > 1:
    cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]
    print(f"[多卡模式] 启用 --tensor-parallel-size {tensor_parallel_size}")
    print(f"[RTX 4090四卡] GPU 0,1,2,3 张量并行 (24GB*4=96GB总显存，90%利用率)")
else:
    print("[单卡模式] 使用RTX 4090 GPU (24GB显存，90%利用率)")

print("\n🎯 RTX 4090四卡张量并行配置 (Qwen3-14B 131K上下文版):")
print(f"  - 📏 序列长度: {MAX_MODEL_LEN} (扩展上下文，支持大规模文档处理)")
print(f"  - 🚀 最大并发数: {MAX_NUM_SEQS} (降低以节省显存)")
print(f"  - 📦 批处理容量: {MAX_NUM_BATCHED_TOKENS} tokens (保守配置)")
print(f"  - 📊 块大小: {BLOCK_SIZE} (符合vLLM要求的16倍数)")
print(f"  - 💾 显存利用率: {GPU_MEMORY_UTILIZATION*100}% (提高以获得更多KV cache)")
print("  - 📐 张量并行: 启用 (四卡分布式)")
print("  - 📋 请求日志: 启用 (便于调试监控)")
print("  - 🧩 分块预填充: 启用")
print("  - 🗄️ 前缀缓存: 关闭 (减少显存占用)")
print("  - 🌀 RoPE缩放: YaRN方法 (factor=4.0, 原始长度=32768)")
print("  - ✅ 预期配置: Qwen3-14B在96GB总显存上运行（四卡张量并行，131K上下文）")

print("\n🔥 正在启动vLLM服务...")
print(f"执行命令: {' '.join(cmd)}")

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ vLLM服务启动失败: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n🛑 服务已停止")