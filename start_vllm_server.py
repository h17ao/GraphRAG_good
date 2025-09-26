#!/usr/bin/env python3

import subprocess
import sys
import os


# MODEL_PATH = ""
MODEL_PATH = ""  # 第3个epoch训练好的模型路径
HOST = "0.0.0.0"
PORT = 8001

GPU_MEMORY_UTILIZATION = 0.85  # 4090 24GB显存利用率 (70%，适应其他进程占用)
MAX_MODEL_LEN = 32768  # 扩展上下文长度，支持大规模文档处理 (32768 × 4)
MAX_NUM_SEQS = 100     # 4090并发数（适中配置）
MAX_NUM_BATCHED_TOKENS = 256  # 4090批处理大小（保守配置）
ENABLE_CHUNKED_PREFILL = True   # 分块预填充（减少峰值显存）
ENABLE_PREFIX_CACHING = False   # 暂时关闭前缀缓存（减少显存占用）
BLOCK_SIZE = 16                 # vLLM要求：块大小必须是16的倍数

os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"


cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if cuda_visible_devices:
    device_list = [d for d in cuda_visible_devices.split(",") if d.strip() != ""]
    tensor_parallel_size = len(device_list)
else:
    tensor_parallel_size = 1


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
    "--served-model-name", "qwen3-14b",
    "--block-size", str(BLOCK_SIZE),         # 显存块大小优化
    "--disable-custom-all-reduce",           # 单卡优化
    "--max-seq-len-to-capture", str(MAX_MODEL_LEN),  # 序列捕获优化
    # "--rope-scaling", '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}',  # RoPE缩放配置
]

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

print("\n🔥 正在启动vLLM服务...")
print(f"执行命令: {' '.join(cmd)}")

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ vLLM服务启动失败: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n🛑 服务已停止")
