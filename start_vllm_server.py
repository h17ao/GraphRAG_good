#!/usr/bin/env python3

import subprocess
import sys
import os


# MODEL_PATH = ""
MODEL_PATH = ""  
HOST = "0.0.0.0"
PORT = 8001

GPU_MEMORY_UTILIZATION = 0.85  
MAX_MODEL_LEN = 32768  
MAX_NUM_SEQS = 100    
MAX_NUM_BATCHED_TOKENS = 256 
ENABLE_CHUNKED_PREFILL = True  
ENABLE_PREFIX_CACHING = False  
BLOCK_SIZE = 16               

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
    "--block-size", str(BLOCK_SIZE),        
    "--disable-custom-all-reduce",         
    "--max-seq-len-to-capture", str(MAX_MODEL_LEN),  
]

if ENABLE_CHUNKED_PREFILL:
    cmd.append("--enable-chunked-prefill")
if ENABLE_PREFIX_CACHING:
    cmd.append("--enable-prefix-caching")
if tensor_parallel_size > 1:
    cmd += ["--tensor-parallel-size", str(tensor_parallel_size)]

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    sys.exit(1)
except KeyboardInterrupt:

