# 禁用日志输出
import logging
import os
logging.getLogger().setLevel(logging.CRITICAL)
os.environ['LOGURU_LEVEL'] = 'CRITICAL'
# 禁用loguru日志文件输出
os.environ['LOGURU_DISABLE'] = 'True'

import argparse
import asyncio
import pandas as pd
from pathlib import Path
import re
import json

from Core.Common.LLM import LLM


def check_sft_format_detailed(answer: str) -> tuple[bool, str]:
    """检查SFT输出格式并返回详细错误信息
    
    支持两种格式：
    1. able格式: <think>...</think><label>able</label><answer>...</answer>
    2. unable格式: <think>...</think><label>unable</label><query>...</query>
    """
    answer = answer.strip()
    
    # 检查是否包含必需的<think>标签
    think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
    if not think_match:
        return False, "缺少<think>标签"
    
    # 检查think标签是否为空
    think_content = think_match.group(1).strip()
    if not think_content:
        return False, "think标签为空"
    
    # 检查是否包含<label>标签
    label_match = re.search(r'<label>\s*(able|unable)\s*</label>', answer, re.DOTALL)
    if not label_match:
        return False, "缺少<label>标签或标签值不是'able'或'unable'"
    
    label_value = label_match.group(1).strip()
    
    # 根据label值检查相应的结束标签
    if label_value == "able":
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        if not answer_match:
            return False, "label为'able'时缺少<answer>标签"
        
        answer_content = answer_match.group(1).strip()
        if not answer_content:
            return False, "answer标签为空"
            
        # 检查标签顺序：think -> label -> answer
        think_pos = think_match.start()
        label_pos = label_match.start()
        answer_pos = answer_match.start()
        
        if not (think_pos < label_pos < answer_pos):
            return False, "标签顺序不正确，应为<think>...<label>able</label>...<answer>..."
        
        # 检查是否有不应该存在的<query>标签
        query_match = re.search(r'<query>(.*?)</query>', answer, re.DOTALL)
        if query_match:
            return False, "label为'able'时不应包含<query>标签"
            
        # 检查是否有标签外的额外文本
        clean_text = answer
        clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<label>\s*able\s*</label>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<answer>.*?</answer>', '', clean_text, flags=re.DOTALL)
        clean_text = clean_text.strip()
        
        if clean_text:
            return False, f"包含标签外的额外文本: {clean_text[:50]}..."
    
    elif label_value == "unable":
        query_match = re.search(r'<query>(.*?)</query>', answer, re.DOTALL)
        if not query_match:
            return False, "label为'unable'时缺少<query>标签"
        
        query_content = query_match.group(1).strip()
        if not query_content:
            return False, "query标签为空"
            
        # 检查标签顺序：think -> label -> query
        think_pos = think_match.start()
        label_pos = label_match.start()
        query_pos = query_match.start()
        
        if not (think_pos < label_pos < query_pos):
            return False, "标签顺序不正确，应为<think>...<label>unable</label>...<query>..."
        
        # 检查是否有不应该存在的<answer>标签
        answer_match = re.search(r'<answer>(.*?)</answer>', answer, re.DOTALL)
        if answer_match:
            return False, "label为'unable'时不应包含<answer>标签"
            
        # 检查是否有标签外的额外文本
        clean_text = answer
        clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<label>\s*unable\s*</label>', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'<query>.*?</query>', '', clean_text, flags=re.DOTALL)
        clean_text = clean_text.strip()
        
        if clean_text:
            return False, f"包含标签外的额外文本: {clean_text[:50]}..."
    
    return True, "格式正确"


def check_sft_format(answer: str) -> bool:
    """检查输出是否遵守SFT格式规则"""
    result, _ = check_sft_format_detailed(answer)
    return result


def log_regenerate(record_id: int, attempt: int, reason: str, log_path: Path):
    """记录重新生成的信息到日志文件"""
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] 记录ID: {record_id}, 重试次数: {attempt}, 原因: {reason}\n"
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(log_entry)


def load_existing_results(output_path: Path):
    """加载已存在的输出 jsonl 文件，返回 (记录列表, 已处理数量)"""
    if output_path.exists():
        try:
            df = pd.read_json(output_path, lines=True)
            records = df.to_dict("records")
            return records, len(records)
        except Exception:
            # 文件损坏或非 jsonl 格式，重新开始
            pass
    return [], 0


def save_results(records, output_path: Path):
    """以 jsonl 格式保存记录列表"""
    df = pd.DataFrame(records)
    df.to_json(output_path, orient="records", lines=True)


# SFT推理prompt模板
SFT_PROMPT_TEMPLATE = """You are a reasoning assistant working with information from a knowledge graph.

Your task is to answer the user query with the provided graph evidence and your own knowledge.
Follow this process for each query:

1. Carefully read the user query and the retrieved graph evidence.
2. Think step-by-step about what the query is asking and what the evidence supports.
3. Use both the graph evidence and your own knowledge to reason through the query.

- If the evidence **is sufficient**, respond in the following format:
  <think> reasoning here. </think>
  <label> able </label>  
  <answer> final answer here. </answer>

- If the evidence **is not sufficient**, do the following:
  - Identify what key information is missing to answer the query.
  - Propose a natural, specific follow-up question that would help obtain the missing evidence.
  - Respond in the following format:   
  <think> reasoning here. </think>
  <label> unable </label>  
  <query> proposed follow-up question here. </query>

Do not include any text outside the specified tags. Maintain strict adherence to the format.

---

User query: {query}

Graph evidence: 
{context_text}"""


async def process_sft_records(records, start_idx: int = 0, max_concurrent: int = 50, output_path: Path = None, existing_records: list = None):
    """处理记录，调用API生成SFT输出"""
    llm = LLM()  # 使用默认配置的 LLM
    all_results = existing_records.copy() if existing_records else []
    dataset_len = len(records)
    regenerate_log_path = Path("regenerate_sft.log")
    
    print(f"🚀 分批并发处理 {dataset_len} 个记录，每批{max_concurrent}个")
    
    # 分批处理
    for batch_start in range(0, len(records), max_concurrent):
        batch_end = min(batch_start + max_concurrent, len(records))
        batch_records = records[batch_start:batch_end]
        
        print(f"📦 处理第 {batch_start//max_concurrent + 1} 批: {batch_start + start_idx + 1} - {batch_end + start_idx}")
        
        # 每批内部并发处理
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _process_single(rec: dict, idx_in_batch: int):
            async with semaphore:
                original_idx = start_idx + batch_start + idx_in_batch
                
                # 构建prompt，使用用户查询和图证据
                query = str(rec.get("query", rec.get("question", "")))
                context_text = str(rec.get("context_text", rec.get("retrieved_context", "")))
                
                prompt = SFT_PROMPT_TEMPLATE.format(
                    query=query,
                    context_text=context_text
                )
                
                max_attempts = 20  # 最多重试20次
                for attempt in range(1, max_attempts + 1):
                    answer = await llm.aask(prompt)
                    
                    is_valid, error_reason = check_sft_format_detailed(answer)
                    if is_valid:
                        # 格式正确，记录完整输出
                        rec["sft_output"] = answer
                        rec["sft_prompt"] = prompt  # 记录发送给大模型的prompt
                        
                        if attempt > 1:
                            print(f"✅ {original_idx + 1}/{start_idx + dataset_len}: 完成 (重试{attempt-1}次后成功)")
                        else:
                            print(f"✅ {original_idx + 1}/{start_idx + dataset_len}: 完成")
                        return rec
                    else:
                        # 格式不正确，记录详细日志
                        reason = f"{error_reason} (第{attempt}次尝试)"
                        log_regenerate(original_idx, attempt, reason, regenerate_log_path)
                        
                        if attempt < max_attempts:
                            print(f"⚠️  {original_idx + 1}/{start_idx + dataset_len}: {error_reason}，重新生成中...")
                        else:
                            print(f"❌ {original_idx + 1}/{start_idx + dataset_len}: 重试{max_attempts}次后仍不符合格式 ({error_reason})，标记为format wrong")
                            # 达到最大重试次数，标记为format wrong
                            rec["sft_output"] = "format wrong"
                            rec["sft_prompt"] = prompt  # 即使格式不正确也记录prompt
                            return rec
                
                return rec  # 理论上不会到达这里
        
        # 执行当前批次的并发查询
        tasks = [_process_single(record, i) for i, record in enumerate(batch_records)]
        batch_results = await asyncio.gather(*tasks)
        
        # 将当前批次结果加入总结果
        all_results.extend(batch_results)
        
        # 每批完成后立即保存
        if output_path:
            save_results(all_results, output_path)
        
        print(f"💾 第 {batch_start//max_concurrent + 1} 批完成，已保存 {len(all_results)} 个记录")
    
    return all_results


async def main():
    parser = argparse.ArgumentParser(description='SFT数据生成器 - 基于图谱证据生成推理输出')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径 (json/jsonl)')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径 (jsonl)')
    parser.add_argument('--max_concurrent', type=int, default=50, help='最大并发数')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    max_concurrent = args.max_concurrent

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件 {input_path} 不存在")

    # 读取输入数据
    try:
        df_input = pd.read_json(input_path, lines=True)
    except ValueError:
        df_input = pd.read_json(input_path)
    input_records = df_input.to_dict("records")

    print(f"[SFT生成] 总记录数: {len(input_records)}")

    # 检查是否有已生成的结果（断点续传）
    existing_records, processed_cnt = load_existing_results(output_path)
    
    if existing_records:
        # 找出已处理的记录（有sft_output字段）
        processed_ids = {rec.get("id") for rec in existing_records if "sft_output" in rec}
        
        # 过滤出未处理的记录
        remaining_records = [rec for rec in input_records if rec.get("id") not in processed_ids]
        
        print(f"[SFT生成] 已处理记录: {len(processed_ids)}, 剩余记录: {len(remaining_records)}")
        
        if not remaining_records:
            print("[SFT生成] 所有记录已处理完成，无需重新处理。")
            return
    else:
        remaining_records = input_records

    if remaining_records:
        # 处理剩余的记录
        final_results = await process_sft_records(
            remaining_records,
            start_idx=len(existing_records) if existing_records else 0,
            max_concurrent=max_concurrent,
            output_path=output_path,
            existing_records=existing_records
        )
        print(f"✅ SFT记录处理完成！共处理 {len(remaining_records)} 个记录")
    else:
        print("[SFT生成] 无需处理记录，输出已是最新版本。")

    print(f"[SFT生成] 结果已保存到 {output_path}")
    print(f"[SFT生成] 重新生成日志已保存到 regenerate_sft.log")


if __name__ == "__main__":
    asyncio.run(main())
