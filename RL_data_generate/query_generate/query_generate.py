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

from Core.Common.LLM import LLM


def check_output_format_detailed(answer: str) -> tuple[bool, str]:
    """检查输出格式并返回详细错误信息"""
    answer = answer.strip()
    
    # 检查是否包含所有必需的标签
    think_match = re.search(r'<think>(.*?)</think>', answer, re.DOTALL)
    query_match = re.search(r'<query>(.*?)</query>', answer, re.DOTALL)
    
    # 必须包含 <think> 标签
    if not think_match:
        return False, "缺少<think>标签"
    
    # 检查think标签是否为空
    think_content = think_match.group(1).strip()
    if not think_content:
        return False, "think标签为空"
    
    # 必须包含 <query> 标签
    if not query_match:
        return False, "缺少<query>标签"
    
    # 检查query标签是否为空
    query_content = query_match.group(1).strip()
    if not query_content:
        return False, "query标签为空"
    
    # 检查标签顺序：think应该在query之前
    think_pos = think_match.start()
    query_pos = query_match.start()
    if think_pos >= query_pos:
        return False, "<think>标签应该在<query>标签之前"
    
    # 检查是否有标签外的额外文本
    clean_text = answer
    clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'<query>.*?</query>', '', clean_text, flags=re.DOTALL)
    clean_text = clean_text.strip()
    
    if clean_text:
        return False, f"包含标签外的额外文本: {clean_text[:50]}..."
    
    return True, "格式正确"


def check_output_format(answer: str) -> bool:
    """检查输出是否遵守prompt中的格式规则"""
    result, _ = check_output_format_detailed(answer)
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


async def process_unable_records(records, prompt_template: str, start_idx: int = 0, max_concurrent: int = 50, output_path: Path = None, existing_records: list = None):
    """处理label为unable的记录，调用API生成查询"""
    llm = LLM()  # 使用默认配置的 LLM
    all_results = existing_records.copy() if existing_records else []
    dataset_len = len(records)
    regenerate_log_path = Path("regenerate_query.log")
    
    print(f"🚀 分批并发处理 {dataset_len} 个unable记录，每批{max_concurrent}个")
    
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
                prompt = (
                    prompt_template.replace("{question}", str(rec.get("question", "")))
                    .replace("{retrieved_context}", str(rec.get("retrieved_context", "")))
                    .replace("{gold_answer}", str(rec.get("gold_answer", "")))
                    .replace("{supporting_facts}", str(rec.get("supporting_facts", "")))
                )
                
                max_attempts = 20  # 最多重试20次
                for attempt in range(1, max_attempts + 1):
                    answer = await llm.aask(prompt)
                    
                    is_valid, error_reason = check_output_format_detailed(answer)
                    if is_valid:
                        # 格式正确，使用这个答案
                        rec["output_final"] = answer
                        rec["prompt"] = prompt  # 记录发送给大模型的prompt
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
                            rec["output_final"] = "format wrong"
                            rec["prompt"] = prompt  # 即使格式不正确也记录prompt
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
    # 硬编码参数
    input_path = Path("hippo_result.json")
    prompt_path = Path("prompt_query_generate.txt")
    output_path = Path("hippo_result_final.json")
    max_concurrent = 50

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template {prompt_path} not found")

    # 读取 prompt 模板
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # 读取输入数据
    try:
        df_input = pd.read_json(input_path, lines=True)
    except ValueError:
        df_input = pd.read_json(input_path)
    input_records = df_input.to_dict("records")

    print(f"[查询生成] 总记录数: {len(input_records)}")

    # 分离able和unable记录
    able_records = [rec for rec in input_records if rec.get("label") == "able"]
    unable_records = [rec for rec in input_records if rec.get("label") == "unable"]
    
    print(f"[查询生成] able记录: {len(able_records)}, unable记录: {len(unable_records)}")

    # 处理able记录：直接复制gold_answer到output_final
    for rec in able_records:
        rec["output_final"] = rec.get("gold_answer", "")
    
    print(f"✅ able记录处理完成，直接复制gold_answer到output_final")

    # 检查是否有已生成的结果（断点续传）
    existing_records, processed_cnt = load_existing_results(output_path)
    
    # 合并able记录和已处理的记录
    all_results = able_records.copy()
    
    if existing_records:
        # 找出已处理的unable记录
        existing_unable = [rec for rec in existing_records if rec.get("label") == "unable" and "output_final" in rec]
        processed_unable_ids = {rec.get("id") for rec in existing_unable}
        
        # 过滤出未处理的unable记录
        remaining_unable = [rec for rec in unable_records if rec.get("id") not in processed_unable_ids]
        
        # 合并已处理的unable记录
        all_results.extend(existing_unable)
        
        print(f"[查询生成] 已处理unable记录: {len(existing_unable)}, 剩余unable记录: {len(remaining_unable)}")
    else:
        remaining_unable = unable_records

    if remaining_unable:
        # 处理剩余的unable记录
        final_results = await process_unable_records(
            remaining_unable,
            prompt_template,
            start_idx=len(able_records) + len([rec for rec in existing_records if rec.get("label") == "unable"]) if existing_records else len(able_records),
            max_concurrent=max_concurrent,
            output_path=output_path,
            existing_records=all_results
        )
        print(f"✅ unable记录处理完成！共处理 {len(remaining_unable)} 个记录")
    else:
        # 保存最终结果
        save_results(all_results, output_path)
        print("[查询生成] 无需处理unable记录，输出已是最新版本。")

    print(f"[查询生成] 结果已保存到 {output_path}")
    print(f"[查询生成] 重新生成日志已保存到 regenerate_query.log")


if __name__ == "__main__":
    asyncio.run(main()) 