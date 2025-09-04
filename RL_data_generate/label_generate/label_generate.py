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
    label_match = re.search(r'<label>\s*(able|unable)\s*</label>', answer, re.IGNORECASE)
    
    # 必须包含 <think> 标签
    if not think_match:
        return False, "缺少<think>标签"
    
    # 检查think标签是否为空
    think_content = think_match.group(1).strip()
    if not think_content:
        return False, "think标签为空"
    
    # 必须包含 <label> 标签，且内容为 "able" 或 "unable"
    if not label_match:
        return False, "缺少<label>标签"
    
    label_content = label_match.group(1).strip().lower()
    if label_content not in ['able', 'unable']:
        return False, f"label内容无效: {label_content}"
    
    # 检查标签顺序：think应该在label之前
    think_pos = think_match.start()
    label_pos = label_match.start()
    if think_pos >= label_pos:
        return False, "<think>标签应该在<label>标签之前"
    
    # 检查是否有标签外的额外文本
    clean_text = answer
    clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'<label>.*?</label>', '', clean_text, flags=re.IGNORECASE)
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


async def generate_answers_with_save(records, prompt_template: str, start_idx: int = 0, max_concurrent: int = 50, output_path: Path = None, existing_records: list = None):
    """分批并发处理记录，每批完成后立即保存"""
    llm = LLM()  # 使用默认配置的 LLM
    all_results = existing_records.copy() if existing_records else []
    dataset_len = len(records)
    regenerate_log_path = Path("regenerate_label.log")
    
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
                prompt = (
                    prompt_template.replace("{question}", str(rec.get("question", "")))
                    .replace("{retrieved_context}", str(rec.get("retrieved_context", "")))
                    .replace("{gold_answer}", str(rec.get("answer", "")))
                )
                
                max_attempts = 100  # 最多重试100次
                for attempt in range(1, max_attempts + 1):
                    answer = await llm.aask(prompt)
                    
                    is_valid, error_reason = check_output_format_detailed(answer)
                    if is_valid:
                        # 格式正确，使用这个答案
                        rec["output_rl"] = answer
                        rec["prompt"] = prompt  # 记录发送给大模型的prompt
                        rec["id"] = original_idx
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
                            rec["output_rl"] = "format wrong"
                            rec["prompt"] = prompt  # 即使格式不正确也记录prompt
                            rec["id"] = original_idx
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
    input_path = Path("hippo_results.json")
    prompt_path = Path("prompt_label_generate.txt")
    output_path = Path("hippo_result_answer.json")
    max_concurrent = 50  # 改为50，与GraphRAG-new保持一致

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} not found")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template {prompt_path} not found")

    # 读取 prompt 模板
    prompt_template = prompt_path.read_text(encoding="utf-8")

    # 读取输入数据（jsonl 或普通 json）
    try:
        df_input = pd.read_json(input_path, lines=True)
    except ValueError:
        df_input = pd.read_json(input_path)
    input_records = df_input.to_dict("records")

    # 如有已生成结果则断点续跑
    existing_records, processed_cnt = load_existing_results(output_path)
    remaining_records = input_records[processed_cnt:]

    print(f"[数据生成] 总记录数: {len(input_records)}, 已处理: {processed_cnt}, 剩余: {len(remaining_records)}")

    if remaining_records:
        # 分批处理，每批完成后立即保存
        final_results = await generate_answers_with_save(
            remaining_records,
            prompt_template,
            start_idx=processed_cnt,
            max_concurrent=max_concurrent,
            output_path=output_path,
            existing_records=existing_records
        )
        print(f"✅ 全部处理完成！共处理 {len(remaining_records)} 个记录")
    else:
        print("[数据生成] 无需处理，输出已是最新版本。")

    print(f"[数据生成] 结果已保存到 {output_path}")
    print(f"[数据生成] 重新生成日志已保存到 regenerate_label.log")


if __name__ == "__main__":
    asyncio.run(main()) 