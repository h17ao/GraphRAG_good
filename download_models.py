#!/usr/bin/env python3
"""
GraphRAG模型下载脚本
- 下载模型 (all-MiniLM-L6-v2, Qwen3-1.7B, DeepSeek-LLM-7B-Chat)
"""

from modelscope import snapshot_download
import os
import sys



def download_model(model_id, local_dir, model_name):
    """下载模型"""
    print(f"\n📥 {model_name}")
    
    try:
        os.makedirs(local_dir, exist_ok=True)
        
        # 检查是否已存在
        if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 5:
            print(f"✅ 已存在，跳过")
            return local_dir
        
        print("⬇️ 下载中...")
        cache_dir = snapshot_download(model_id=model_id, local_dir=local_dir, revision='master')
        print(f"✅ 完成")
        return cache_dir
        
    except Exception as e:
        print(f"❌ 失败: {str(e)}")
        return None

def main():
    """主函数"""
    print("🚀 GraphRAG模型下载")
    
    # 模型配置
    models = [
        ("AI-ModelScope/all-MiniLM-L6-v2", "../cache/models/modelscope/hub/models/AI-ModelScope/all-MiniLM-L6-v2", "嵌入模型"),
        # ("Qwen/Qwen3-8B", "../cache/models/modelscope/hub/models/qwen/Qwen3-8B", "Qwen3-8B"),
        # ("Qwen/Qwen3-4B", "../cache/models/modelscope/hub/models/qwen/Qwen3-4B", "Qwen3-4B"),
        # ("Qwen/Qwen3-32B", "../cache/models/modelscope/hub/models/qwen/Qwen3-32B", "Qwen3-32B")
        ("Qwen/Qwen3-8B", "../cache/models/modelscope/hub/models/qwen/Qwen3-8B", "Qwen3-8B"),
        # ("Qwen/Qwen3-0.6B", "../cache/models/modelscope/hub/models/qwen/Qwen3-0.6B", "Qwen3-0.6B")
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "../cache/models/modelscope/hub/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama-8B")
    ]
    
    success_count = 0
    
    # 下载模型
    for model_id, local_dir, name in models:
        if download_model(model_id, local_dir, name):
            success_count += 1
    
    # 最终总结
    print(f"\n🎉 模型下载完成: {success_count}/{len(models)}")
    if success_count == len(models):
        print("✅ 所有模型已准备就绪!")
    else:
        print("⚠️ 部分模型下载失败，请检查网络连接")
        sys.exit(1)
    
if __name__ == "__main__":
    main() 