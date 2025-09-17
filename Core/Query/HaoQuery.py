from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from Core.Prompt.HaoPrompt import (
    REASONING_STEP_PROMPT_QWEN, FORCE_ANSWER_PROMPT_QWEN,
    REASONING_STEP_PROMPT, FORCE_ANSWER_PROMPT,
    NO_CONTEXT_AVAILABLE
)
from Core.Common.Utils import truncate_str_by_token_size
import re
import json
import pandas as pd
from io import StringIO
import os


class HaoQuery(BaseQuery):
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)
        self.max_iterations = getattr(config, 'max_iterations', 5)  # 默认最大迭代5次
        self.retrieval_method = getattr(config, 'retrieval_method', 'ppr')  # 默认使用PPR
        self.all_model_responses = []  # 新增：记录所有模型回复
        

                    
        # 根据检索模型类型选择prompt模板
        self._set_prompt_templates()
        
        # 初始化具体的检索方法实例
        self._init_retrieval_method(config, retriever_context)
        
        # 配置上下文token限制 - 基于检索模型的MAX_MODEL_LEN计算
        model_max_len = getattr(self.llm.config, 'MAX_MODEL_LEN', None)  # 获取检索模型的最大长度
        if model_max_len is None:
            model_max_len = 128000  # 默认值，适用于大多数现代LLM
        max_token = getattr(self.llm.config, 'max_token', None)  # 获取最大生成token数
        if max_token is None:
            max_token = 28000  # 默认值，适用于大多数LLM
        calculated_max_tokens = model_max_len - max_token - 3000  # MAX_MODEL_LEN - max_token - 3000
        self.max_context_tokens = getattr(config, 'max_context_tokens', calculated_max_tokens)
        logger.info(f"HaoQuery上下文token限制: {self.max_context_tokens} (模型MAX_MODEL_LEN: {model_max_len}, max_token: {max_token})")
        
        # 初始化截断统计信息
        self.total_truncation_time = 0.0  # 累计截断耗时
        self.truncation_count = 0  # 截断次数
        



    
    def _set_prompt_templates(self):
        """根据检索模型类型设置prompt模板"""
        model_name = self.llm.config.model.lower() if hasattr(self.llm.config, 'model') else ""
        
        if 'qwen' in model_name:
            # 使用qwen专用模板
            self.reasoning_prompt = REASONING_STEP_PROMPT_QWEN
            self.force_answer_prompt = FORCE_ANSWER_PROMPT_QWEN
            logger.info(f"检测到qwen模型({model_name})，使用qwen专用prompt模板")
        else:
            # 使用通用模板
            self.reasoning_prompt = REASONING_STEP_PROMPT
            self.force_answer_prompt = FORCE_ANSWER_PROMPT
            logger.info(f"检测到非qwen模型({model_name})，使用通用prompt模板")
    
    def _init_retrieval_method(self, config, retriever_context):
        """初始化具体的检索方法实例"""
        from Core.Query.QueryFactory import QueryFactory
        
        # 创建对应的查询方法实例，用于调用其检索逻辑
        factory = QueryFactory()
        try:
            self.retrieval_query = factory.get_query(self.retrieval_method, config, retriever_context)
            logger.info(f"HaoQuery使用 {self.retrieval_method} 方法进行检索")
        except KeyError:
            logger.error(f"未知的检索方法: {self.retrieval_method}")
            raise ValueError(f"不支持的检索方法: {self.retrieval_method}")


    
    def _parse_response(self, response, extract_answer_only=False):
        """解析LLM回复，提取标签内容"""
        if extract_answer_only:
            # 只提取answer标签（格式已验证过）
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            return match.group(1).strip()
        
        # 提取完整的推理回复（格式已验证过）
        think = self._extract_tag(response, 'think')
        label = self._extract_tag(response, 'label')
        
        if label.lower() == "able":
            content = self._extract_tag(response, 'answer')
        elif label.lower() == "unable":
            content = self._extract_tag(response, 'query')
        else:
            content = ""
            
        return think, label, content
    
    def _extract_tag(self, text, tag):
        """提取指定标签的内容"""
        match = re.search(f'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _truncate_contexts(self, all_contexts):
        """截断累积的上下文，确保不超过token限制"""
        if not all_contexts:
            return NO_CONTEXT_AVAILABLE
        
        # 从TextChunk对象或字符串中提取文本内容
        context_strings = []
        for context in all_contexts:
            if hasattr(context, 'content'):
                # TextChunk对象，提取content
                context_strings.append(context.content)
            else:
                # 字符串格式
                context_strings.append(str(context))
            
        # 合并所有上下文并去重
        context_text = "\n\n".join(list(dict.fromkeys(context_strings)))
        
        # 如果需要，进行token截断
        if self.max_context_tokens > 0:
            import time
            from Core.Common.Utils import encode_string_by_tiktoken
            
            # 记录原始token数
            start_time = time.time()
            original_tokens = len(encode_string_by_tiktoken(context_text))
            
            truncated_text = truncate_str_by_token_size(context_text, self.max_context_tokens)
            
            # 记录截断后token数和耗时
            if truncated_text != context_text:
                truncated_tokens = len(encode_string_by_tiktoken(truncated_text))
                elapsed_time = time.time() - start_time
                
                # 更新累计统计
                self.truncation_count += 1
                self.total_truncation_time += elapsed_time
                
                logger.info(f"上下文截断: {original_tokens} tokens -> {truncated_tokens} tokens，累计耗时: {self.total_truncation_time:.3f}s")
            return truncated_text
        
        return context_text

    def _is_valid_reasoning_format(self, response):
        """验证推理响应格式是否正确"""
        try:
            # 检查是否包含必需的标签
            has_think = bool(re.search(r'<think>(.*?)</think>', response, re.DOTALL))
            has_label = bool(re.search(r'<label>(.*?)</label>', response, re.DOTALL))
            
            if not (has_think and has_label):
                return False
                
            # 检查label的值是否有效
            label_match = re.search(r'<label>(.*?)</label>', response, re.DOTALL)
            if label_match:
                label = label_match.group(1).strip().lower()
                if label == "able":
                    # 如果是able，必须有answer标签
                    return bool(re.search(r'<answer>(.*?)</answer>', response, re.DOTALL))
                elif label == "unable":
                    # 如果是unable，必须有query标签
                    return bool(re.search(r'<query>(.*?)</query>', response, re.DOTALL))
                else:
                    return False
            
            return False
        except Exception:
            return False

    def _is_valid_force_answer_format(self, response):
        """验证强制作答响应格式是否正确"""
        try:
            # 检查是否包含必需的标签
            has_think = bool(re.search(r'<think>(.*?)</think>', response, re.DOTALL))
            has_answer = bool(re.search(r'<answer>(.*?)</answer>', response, re.DOTALL))
            
            # 强制作答必须同时包含think和answer标签
            return has_think and has_answer
        except Exception:
            return False

    async def _reasoning_step(self, query, all_contexts, previous_queries=None, current_iteration=1, local_model_responses=None):
        """使用推理助手判断当前累积的所有context是否足够回答query"""
        # 构建历史查询信息
        history_text = ""
        if previous_queries:
            history_text = f"\nPrevious queries (AVOID repeating):\n" + "\n".join([f"- {pq}" for pq in previous_queries]) + "\n"

        # 构建并截断上下文文本
        context_text = self._truncate_contexts(all_contexts)
        
        # 使用根据模型类型选择的prompt模板
        prompt = self.reasoning_prompt.format(
            query=query,
            context_text=context_text
        )

        # 传统推理路径
        logger.info(f"第 {current_iteration} 次迭代: 使用传统推理方法")

        # 重试逻辑：最多尝试3次
        last_response = ""
        for attempt in range(3):
            try:
                logger.debug(f"推理步骤第 {attempt + 1}/3 次尝试")
                response = await self.llm.aask(msg=prompt)
                last_response = response
                
                # 调试: 打印传统LLM的推理回复
                logger.info(f"🔍 传统LLM推理回复（第{attempt + 1}次尝试）:\n{response}")
                
                # 验证格式是否正确
                if self._is_valid_reasoning_format(response):
                    logger.debug(f"第 {attempt + 1} 次尝试格式正确")
                    
                    # 记录成功的attempt
                    if local_model_responses is not None:
                        local_model_responses.append({
                            "step": "reasoning",
                            "iteration": current_iteration,
                            "attempt": attempt + 1,
                            "query": query,
                            "response": response
                        })
                    
                    return self._parse_response(response)
                else:
                    logger.warning(f"第 {attempt + 1} 次尝试格式不正确，准备重试")
                        
            except Exception as e:
                logger.warning(f"推理步骤第 {attempt + 1} 次尝试出错: {e}，准备重试")
                last_response = str(e)
        
        # 3次尝试都失败，记录最后一次尝试
        logger.error("3次尝试后仍无法获得正确格式，返回空字符串")
        if local_model_responses is not None:
            local_model_responses.append({
                "step": "reasoning",
                "iteration": current_iteration,
                "attempt": 3,
                "query": query,
                "response": ""
            })
        return ("", "able", "")



    async def _retrieve_relevant_contexts(self, query):
        """
        使用配置的检索方法获取相关上下文
        
        Args:
            query: 查询文本
            
        Returns:
            返回检索到的contexts列表
        """
        logger.debug(f"使用 {self.retrieval_method} 方法检索: {query}")
        
        # 调用传统的检索逻辑
        contexts = await self.retrieval_query._retrieve_relevant_contexts(query)
        formatted_contexts = self._format_contexts(contexts)
        
        return formatted_contexts
    
    def _format_contexts(self, contexts):
        """统一格式化上下文返回格式"""
        if contexts is None:
            return []
        elif isinstance(contexts, str):
            return [contexts]  # 某些方法可能返回字符串
        elif isinstance(contexts, (list, tuple)):
            # 直接返回列表，保持TextChunk对象的完整性
            return list(contexts)
        else:
            logger.warning(f"检索方法 {self.retrieval_method} 返回了意外的格式: {type(contexts)}")
            return [str(contexts)]

    async def query(self, query):
        """重写query方法，实现迭代查询逻辑"""
        # 使用局部变量而不是实例变量，避免并发竞争
        local_model_responses = []
        
        # 获取初始上下文
        initial_context = await self._retrieve_relevant_contexts(query)
        
        # 执行迭代查询
        result = await self.generation_qa(query, initial_context, local_model_responses)
        
        # 返回兼容格式和核心信息
        final_answer = result.get("final_answer", "")
        all_contexts = result.get("all_contexts", [])
        
        # 记录所有查询的截断统计总结
        if self.truncation_count > 0:
            logger.info(f"查询完成 - 所有查询累计进行 {self.truncation_count} 次截断, 累计截断耗时: {self.total_truncation_time:.3f}s")
        
        return {
            "response": final_answer,  # 保持兼容性 - main.py用于output字段
            "context": all_contexts,   # 保持兼容性 - main.py用于retrieved_context字段
            "total_iterations": result.get("total_iterations", 1),
            "historical_queries": result.get("historical_queries", []),
            "output_all": local_model_responses # 新增：添加所有模型回复
        }

    async def generation_qa(self, query, context, local_model_responses):
        """迭代查询的主要逻辑"""
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
              
        current_query = query
        all_contexts = []
        previous_queries = []  # 用于传递给LLM，避免重复失败的查询
        all_executed_queries = []  # 记录所有执行过的查询
        current_iteration = 0
        
        logger.info(f"开始HaoQuery迭代查询: {query}")
        
        for iteration in range(self.max_iterations):
            current_iteration = iteration + 1
            logger.info(f"=== 迭代 {current_iteration}/{self.max_iterations}: {current_query} ===")
            
            # 记录当前执行的查询
            all_executed_queries.append(current_query)
            
            # 获取上下文
            if iteration == 0:
                current_context = context
            else:
                current_context = await self._retrieve_relevant_contexts(current_query)
            all_contexts.extend(current_context if current_context else [])
            
            # 最后一次迭代：强制作答
            if iteration == self.max_iterations - 1:
                logger.info(f"达到最大迭代数，强制作答")
                final_answer = await self._force_answer(query, all_contexts, current_iteration, local_model_responses)
                logger.info(f"强制作答完成")
                
                return self._build_result(final_answer, current_iteration, all_contexts, all_executed_queries)
            
            # 推理判断
            if previous_queries:
                logger.debug(f"历史查询: {previous_queries}")
                
            think, label, content = await self._reasoning_step(query, all_contexts, previous_queries, current_iteration, local_model_responses)
            
            if label.lower() == "able":
                logger.info(f"第 {current_iteration} 次迭代找到答案")
                return self._build_result(content, current_iteration, all_contexts, all_executed_queries)
                
            elif label.lower() == "unable" and content:
                logger.debug(f"建议新查询: {content}")
                previous_queries.append(current_query)  # 只有失败的查询加入previous_queries
                current_query = content
            else:
                logger.warning(f"第 {current_iteration} 次迭代无效")
                break
        
        # 异常退出
        logger.info(f"异常退出，强制作答")
        final_answer = await self._force_answer(query, all_contexts, current_iteration, local_model_responses)
        return self._build_result(final_answer, current_iteration, all_contexts, all_executed_queries)
    
    def _build_result(self, final_answer, total_iterations, all_contexts, historical_queries):
        """构建返回结果"""
        return {
            "final_answer": final_answer,
            "total_iterations": total_iterations,
            "all_contexts": all_contexts,
            "historical_queries": historical_queries
        }

    async def _force_answer(self, query, all_contexts, current_iteration=None, local_model_responses=None):
        """强制作答模式：使用累积的所有上下文让LLM直接回答"""
        # 准备并截断上下文文本
        context_text = self._truncate_contexts(all_contexts)
        
        # 使用根据模型类型选择的prompt模板
        prompt = self.force_answer_prompt.format(
            query=query,
            context_text=context_text
        )

        # 传统强制作答路径
        logger.info("使用传统强制作答方法")

        # 重试逻辑：最多尝试3次
        last_response = ""
        for attempt in range(3):
            try:
                logger.debug(f"强制作答第 {attempt + 1}/3 次尝试")
                response = await self.llm.aask(msg=prompt, system_msgs=["You are a helpful assistant."])
                last_response = response
                
                # 调试: 打印传统LLM的强制作答回复
                logger.info(f"🔍 传统LLM强制作答回复（第{attempt + 1}次尝试）:\n{response}")
                
                # 验证格式是否正确
                if self._is_valid_force_answer_format(response):
                    logger.debug(f"第 {attempt + 1} 次尝试格式正确")
                    
                    # 记录成功的attempt
                    if local_model_responses is not None:
                        local_model_responses.append({
                            "step": "force_answer",
                            "iteration": current_iteration if current_iteration is not None else self.max_iterations,
                            "attempt": attempt + 1,
                            "query": query,
                            "response": response
                        })
                    
                    return self._parse_response(response, extract_answer_only=True)
                else:
                    logger.warning(f"第 {attempt + 1} 次尝试格式不正确，准备重试")
                    
            except Exception as e:
                logger.warning(f"强制作答第 {attempt + 1} 次尝试出错: {e}，准备重试")
                last_response = str(e)
        
        # 3次尝试都失败，记录最后一次尝试
        logger.error("强制作答3次尝试后仍无法获得正确格式，返回空字符串")
        if local_model_responses is not None:
            local_model_responses.append({
                "step": "force_answer",
                "iteration": current_iteration if current_iteration is not None else self.max_iterations,
                "attempt": 3,
                "query": query,
                "response": ""
            })
        return ""



    async def generation_summary(self, query, context):
        """HaoQuery专注于QA任务，不支持总结功能"""
        return "HaoQuery专注于问答任务，不支持总结功能"