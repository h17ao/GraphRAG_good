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
        if config is not None:
            context = Context()
            context.config = config  
            self.llm = EvalLLM(context)
        elif llm_api_key and base_url and model:
            llm_config = LLMConfig(
                api_key=llm_api_key,
                base_url=base_url,
                model=model,
                temperature=0.3,
                calc_usage=True  
            )
            self.llm = LLM(llm_config)
        else:
            raise ValueError("必须提供config参数或者(llm_api_key, base_url, model)参数")


        self.template = """
Given the question and gold answer, please judge if the predicted answer is correct. Providing additional background or supplementary information is acceptable.

Question: {question}.

Gold Answer: {gold_answer}.

Predicted Answer: {answer}.
        
Please only return 1 (means correct) or 0 (means incorrect) in a concise way.
"""
        
    def extract_content_outside_think_tags(self, text):

        if not isinstance(text, str):
            text = str(text)
        
        pattern = r'<think>.*?</think>'

        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text.strip())
        return cleaned_text

    async def eval_single_item(self, question, gold_answer, answer):

        prompt = self.template.format(question=question, gold_answer=gold_answer, answer=answer)
        
        max_attempts = 3  # 最多重试3次
        
        for attempt in range(1, max_attempts + 1):
            try:
                # 使用项目的LLM系统，这样会自动记录成本
                raw_response = await self.llm.aask(prompt)
                # 确保raw_response是字符串类型
                if not isinstance(raw_response, str):
                    raw_response = str(raw_response)
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
                    retry_interval = 1.0  # 重试间隔1秒
                    await asyncio.sleep(retry_interval)
                    continue
                else:
                    logger.error(f"达到最大重试次数{max_attempts}，评估出错: {e}，默认返回0")
                    return 0, f"Error: {str(e)}"

        return 0, "Maximum attempts reached"

    async def evaluate(self, results_path):
        df = pd.read_json(results_path, lines=True)
        print(f"开始LLM评估 {len(df)} 条记录")
        
        semaphore = asyncio.Semaphore(50)
        start_time = time.time()
        
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
        

        final_costs = self.llm.get_costs()
        eval_prompt_tokens = final_costs.total_prompt_tokens - initial_costs.total_prompt_tokens
        eval_completion_tokens = final_costs.total_completion_tokens - initial_costs.total_completion_tokens
        eval_cost = final_costs.total_cost - initial_costs.total_cost
        
        logger.info(f"LLM评估成本统计: 模型: {final_costs.model_name} | prompt_tokens: {eval_prompt_tokens}, completion_tokens: {eval_completion_tokens}, cost: ${eval_cost:.4f}")
        

        results_dir = Path(results_path).parent
        

        from datetime import datetime
        timestamp = datetime.now().strftime("%m%d%H%M")
        
        df.to_json(results_dir / f"llm_eval_results_{timestamp}.json", orient="records", lines=True)
        

        incorrect_count = len(df) - correct_count  
        eval_failure_count = incorrect_count - api_failure_count 
        
        summary = {
            "total_count": len(df),
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "eval_failure_count": eval_failure_count,  
            "api_failure_count": api_failure_count,    
            "accuracy": accuracy,
            "total_time": total_time,
            "avg_time_per_item": total_time / len(df),
            "eval_model": final_costs.model_name,  
            "eval_prompt_tokens": eval_prompt_tokens,
            "eval_completion_tokens": eval_completion_tokens,
            "eval_cost": eval_cost
        }
        
        with open(results_dir / f"llm_eval_summary_{timestamp}.json", "w") as f:
            json.dump({"summary": summary}, f, indent=2)



async def wrapper_llm_evaluation(path, api_key=None, base_url=None, model=None, config=None):
    if config is not None:
        evaluator = LLMEvaluator(config=config)
    else:
        evaluator = LLMEvaluator(api_key, base_url, model)
    await evaluator.evaluate(path) 
