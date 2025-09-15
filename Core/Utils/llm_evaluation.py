import time
import json
import re
import asyncio
import pandas as pd
from pathlib import Path
from tenacity import RetryError
from Core.Common.Logger import logger
from Core.Common.LLM import LLM, EvalLLM
from Core.Common.Context import Context
from Config.LLMConfig import LLMConfig


class LLMEvaluator:
    def __init__(self, llm_api_key=None, base_url=None, model=None, config=None):
        # 支持两种初始化方式：
        # 1. 传统方式：直接传入API参数
        # 2. 新方式：使用config中的eval_llm配置
        if config is not None:
            # 使用config中的eval_llm配置
            context = Context()
            context.config = config  # 直接赋值而不是使用set_config
            self.llm = EvalLLM(context)
        elif llm_api_key and base_url and model:
            # 使用传统的API参数方式
            llm_config = LLMConfig(
                api_key=llm_api_key,
                base_url=base_url,
                model=model,
                temperature=0.3,
                calc_usage=True  # 启用成本计算
            )
            self.llm = LLM(llm_config)
        else:
            raise ValueError("必须提供config参数或者(llm_api_key, base_url, model)参数")

        # 评估模板
        self.template = """
Given the question and gold answer, please judge if the predicted answer is correct. Providing additional background or supplementary information is acceptable.

Question: {question}.

Gold Answer: {gold_answer}.

Predicted Answer: {answer}.
        
Please only return 1 (means correct) or 0 (means incorrect) in a concise way.
"""
        
    def extract_content_outside_think_tags(self, text):
        """提取<think></think>标签以外的内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 去除<think></think>标签及其内容后的文本
        """
        # 使用正则表达式匹配<think>...</think>标签（支持多行）
        pattern = r'<think>.*?</think>'
        # 使用re.DOTALL标志使.匹配包括换行符在内的任何字符
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        # 清理多余的空白字符
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text.strip())
        return cleaned_text

    async def eval_single_item(self, question, gold_answer, answer):
        """评估单个问题的答案正确性
        
        Returns:
            tuple: (eval_score, response) - eval_score为0或1，response为评估推理过程
        """
        prompt = self.template.format(question=question, gold_answer=gold_answer, answer=answer)
        
        max_attempts = 3  # 最多重试3次
        
        for attempt in range(1, max_attempts + 1):
            try:
                # 使用项目的LLM系统，这样会自动记录成本
                raw_response = await self.llm.aask(prompt)
                # 提取<think></think>标签以外的内容
                response_text = self.extract_content_outside_think_tags(raw_response).strip()
                
                # 直接提取第一个字符来判断是0还是1
                if response_text and len(response_text) > 0:
                    first_char = response_text[0]
                    if first_char == '1':
                        eval_score = 1
                        return eval_score, response_text
                    elif first_char == '0':
                        eval_score = 0
                        return eval_score, response_text
                    else:
                        # 第一个字符不是0或1，继续重试
                        if attempt < max_attempts:
                            logger.warning(f"响应的第一个字符必须是'1'或'0'，当前是: '{first_char}'，第{attempt}次重试")
                            await asyncio.sleep(1)
                            continue
                        else:
                            logger.error(f"达到最大重试次数，第一个字符仍无效: '{first_char}'，默认返回0")
                            return 0, response_text
                else:
                    # 响应为空，需要重试
                    if attempt < max_attempts:
                        logger.warning(f"响应为空，第{attempt}次重试")
                        await asyncio.sleep(1)
                        continue
                    else:
                        logger.error(f"达到最大重试次数{max_attempts}，响应仍为空，默认返回0")
                        return 0, response_text or "Empty response"
                        
            except RetryError as e:
                # 捕获底层API的RetryError，直接返回0并继续评估
                logger.error(f"底层API重试失败，已达到最大重试次数: {e}，返回评估分数0")
                return 0, f"API Retry Error: {str(e)}"
            except Exception as e:
                if attempt < max_attempts:
                    logger.warning(f"评估出错: {e}，第{attempt}/{max_attempts}次重试")
                    await asyncio.sleep(retry_interval)
                    continue
                else:
                    logger.error(f"达到最大重试次数{max_attempts}，评估出错: {e}，默认返回0")
                    return 0, f"Error: {str(e)}"
        
        # 理论上不会到达这里，但为了安全起见
        return 0, "Maximum attempts reached"

    async def evaluate(self, results_path):
        df = pd.read_json(results_path, lines=True)
        print(f"开始LLM评估 {len(df)} 条记录")
        
        semaphore = asyncio.Semaphore(50)
        start_time = time.time()
        
        # 记录评估阶段开始时的成本
        initial_costs = self.llm.get_costs()
        
        async def process_item(idx, row):
            async with semaphore:
                eval_score, eval_response = await self.eval_single_item(
                    row.get("question", ""),
                    row.get("answer", ""), 
                    row.get("output", "")
                )
                return idx, eval_score, eval_response
        
        tasks = [process_item(idx, row) for idx, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        
        correct_count = 0
        api_failure_count = 0  # 统计API失败导致的0分记录
        for idx, eval_score, eval_response in results:
            df.loc[idx, "llm_eval_correct"] = eval_score
            df.loc[idx, "llm_eval_response"] = eval_response  # 记录完整的LLM响应
            if eval_score == 1:
                correct_count += 1
            elif "API Retry Error:" in str(eval_response):
                api_failure_count += 1
        
        total_time = time.time() - start_time
        accuracy = correct_count / len(df)
        
        # 计算评估阶段的成本
        final_costs = self.llm.get_costs()
        eval_prompt_tokens = final_costs.total_prompt_tokens - initial_costs.total_prompt_tokens
        eval_completion_tokens = final_costs.total_completion_tokens - initial_costs.total_completion_tokens
        eval_cost = final_costs.total_cost - initial_costs.total_cost
        
        logger.info(f"LLM评估成本统计: 模型: {final_costs.model_name} | prompt_tokens: {eval_prompt_tokens}, completion_tokens: {eval_completion_tokens}, cost: ${eval_cost:.4f}")
        
        # 保存结果
        results_dir = Path(results_path).parent
        
        # 生成时间戳文件名
        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d%H%M")
        
        df.to_json(results_dir / f"llm_eval_results_{timestamp}.json", orient="records", lines=True)
        
        # 计算各类统计数据
        incorrect_count = len(df) - correct_count  # 总的错误数量
        eval_failure_count = incorrect_count - api_failure_count  # 评估判定为错误的数量（排除API失败）
        
        summary = {
            "total_count": len(df),
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "eval_failure_count": eval_failure_count,  # 评估判定为错误的数量
            "api_failure_count": api_failure_count,    # API失败导致的0分数量
            "accuracy": accuracy,
            "total_time": total_time,
            "avg_time_per_item": total_time / len(df),
            "eval_model": final_costs.model_name,  # 添加评估模型名称
            "eval_prompt_tokens": eval_prompt_tokens,
            "eval_completion_tokens": eval_completion_tokens,
            "eval_cost": eval_cost
        }
        
        with open(results_dir / f"llm_eval_summary_{timestamp}.json", "w") as f:
            json.dump({"summary": summary}, f, indent=2)
        
        print(f"LLM评估完成: 准确率 {accuracy:.4f}, 耗时 {total_time:.2f}秒")
        print(f"统计详情: 总计{len(df)}条, 正确{correct_count}条, 评估错误{eval_failure_count}条, API失败{api_failure_count}条")


async def wrapper_llm_evaluation(path, api_key=None, base_url=None, model=None, config=None):
    """LLM评估的包装函数
    
    Args:
        path: 结果文件路径
        api_key: LLM API密钥 (传统方式)
        base_url: LLM基础URL (传统方式)  
        model: LLM模型名称 (传统方式)
        config: 配置对象，包含eval_llm设置 (新方式)
    """
    if config is not None:
        # 使用新的config方式
        evaluator = LLMEvaluator(config=config)
    else:
        # 使用传统的API参数方式
        evaluator = LLMEvaluator(api_key, base_url, model)
    await evaluator.evaluate(path) 