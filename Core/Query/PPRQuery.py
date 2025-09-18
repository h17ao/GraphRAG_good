from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from Core.Common.Memory import Memory
from Core.Schema.Message import Message
import threading



class PPRQuery(BaseQuery):
    # 类级别的线程安全计数器
    _no_second_retrieval_questions = 0
    _lock = threading.Lock()
    
    def __init__(self, config, retriever_context):
        super().__init__(config, retriever_context)
        
        # 配置上下文token限制 - 基于检索模型的MAX_MODEL_LEN计算
        model_max_len = getattr(self.llm.config, 'MAX_MODEL_LEN', None)
        max_token = getattr(self.llm.config, 'max_token', None)
        
        if model_max_len is None:
            raise ValueError(f"检索LLM配置中缺少MAX_MODEL_LEN参数，请在Config2.yaml的retrieval_llm部分设置MAX_MODEL_LEN")
        if max_token is None:
            raise ValueError(f"检索LLM配置中缺少max_token参数，请在Config2.yaml的retrieval_llm部分设置max_token")
            
        calculated_max_tokens = model_max_len - max_token - 3000
        self.max_context_tokens = getattr(config, 'max_context_tokens', calculated_max_tokens)
        logger.info(f"PPRQuery上下文token限制: {self.max_context_tokens} (模型MAX_MODEL_LEN: {model_max_len}, max_token: {max_token})")

    def _truncate_context(self, context_text):
        """截断上下文文本以符合token限制"""
        if self.max_context_tokens <= 0:
            return context_text
            
        from Core.Common.Utils import truncate_str_by_token_size, encode_string_by_tiktoken
        import time
        
        start_time = time.time()
        original_tokens = len(encode_string_by_tiktoken(context_text))
        
        truncated_text = truncate_str_by_token_size(context_text, self.max_context_tokens)
        
        if truncated_text != context_text:
            truncated_tokens = len(encode_string_by_tiktoken(truncated_text))
            elapsed_time = time.time() - start_time
            logger.info(f"PPRQuery上下文截断: {original_tokens} tokens -> {truncated_tokens} tokens，耗时: {elapsed_time:.3f}s")
            
            # 安全检查：确保截断后的token数不超过模型限制
            model_max_len = getattr(self.llm.config, 'MAX_MODEL_LEN', 32768)
            max_token = getattr(self.llm.config, 'max_token', 6144)
            safe_limit = model_max_len - max_token - 3000  # 额外3000token的安全边距
            
            if truncated_tokens > safe_limit:
                logger.warning(f"截断后token数({truncated_tokens})仍超过安全限制({safe_limit})，进行二次截断")
                truncated_text = truncate_str_by_token_size(truncated_text, safe_limit)
                truncated_tokens = len(encode_string_by_tiktoken(truncated_text))
                logger.info(f"二次截断后: {truncated_tokens} tokens")
        
        return truncated_text

    async def reason_step(self, few_shot: list, query: str, passages: list, thoughts: list):
        """
        Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
        :return: next thought
        """
        prompt_demo = ''
        for sample in few_shot:
            prompt_demo += f'{sample["document"]}\n\nQuestion: {sample["question"]}\nThought: {sample["answer"]}\n\n'

        prompt_user = ''

        # TODO: merge title for the hotpotQA dataset
        for passage in passages:
            # 处理TextChunk对象，提取content内容
            if hasattr(passage, 'content'):
                prompt_user += f'{passage.content}\n\n'
            else:
                # 兼容字符串格式
                prompt_user += f'{passage}\n\n'
        prompt_user += f'Question: {query} \n Thought:' + ' '.join(thoughts)

        try:
            response_content = await self.llm.aask(msg=prompt_demo + prompt_user,
                                                   system_msgs=[QueryPrompt.IRCOT_REASON_INSTRUCTION])

        except Exception as e:
            print(e)
            return ''
        return response_content

    async def _retrieve_relevant_contexts(self, query):
        """
        检索相关上下文
        
        Args:
            query: 查询文本
            
        Returns:
            返回检索到的passages列表
        """
        # NOTE: link-entity and extract entity are different

        entities = await self.extract_query_entities(query)
        if not self.config.augmentation_ppr:
            # For HippoRAG 
            retrieved_passages, scores = await self._retriever.retrieve_relevant_content(query=query,
                                                                                         seed_entities=entities,
                                                                                         link_entity=True,
                                                                                         type=Retriever.CHUNK,
                                                                                         mode="ppr")
            thoughts = []
            # 使用chunk_id作为key来跟踪TextChunk对象的分数
            passage_scores = {}
            chunk_id_to_passage = {}
            for passage, score in zip(retrieved_passages, scores):
                if hasattr(passage, 'chunk_id'):
                    chunk_id = passage.chunk_id
                    passage_scores[chunk_id] = score
                    chunk_id_to_passage[chunk_id] = passage
                else:
                    # 回退方案：对于字符串，使用内容作为key（截断以避免太长）
                    key = str(passage)[:100]  # 使用前100个字符作为key
                    passage_scores[key] = score
                    chunk_id_to_passage[key] = passage
            few_shot_examples = []
            
            # Iterative refinement loop
            for iteration in range(2, self.config.max_ir_steps + 1):
                logger.info("Entering the ir-cot iteration: {}".format(iteration))
                # Generate a new thought based on current passages and thoughts
                new_thought = await self.reason_step(few_shot_examples, query, retrieved_passages[: self.config.top_k],
                                                     thoughts)
                thoughts.append(new_thought)

                # Check if the thought contains the answer
                if 'So the answer is:' in new_thought:
                    if iteration == 2:  # 第一次迭代就找到答案
                        with PPRQuery._lock:
                            PPRQuery._no_second_retrieval_questions += 1
                            count = PPRQuery._no_second_retrieval_questions
                        print(f"hippo没有二次推理检索 (累计: {count})")
                    break
                
                # Retrieve new passages based on the new thought
                new_passages, new_scores = await self._retriever.retrieve_relevant_content(query=query,
                                                                                           seed_entities=thoughts,
                                                                                            link_entity=True,
                                                                                           type=Retriever.CHUNK,
                                                                                           mode="ppr")

                # Update passage scores
                for passage, score in zip(new_passages, new_scores):
                    if hasattr(passage, 'chunk_id'):
                        key = passage.chunk_id
                    else:
                        key = str(passage)[:100]
                    
                    if key in passage_scores:
                        passage_scores[key] = max(passage_scores[key], score)
                    else:
                        passage_scores[key] = score
                        chunk_id_to_passage[key] = passage

                # Sort passages by score in descending order
                sorted_items = sorted(
                    passage_scores.items(), key=lambda item: item[1], reverse=True
                )
                retrieved_passages = [chunk_id_to_passage[key] for key, score in sorted_items]
                scores = [score for key, score in sorted_items]

            return retrieved_passages

        else:
            retrieved_passages = await self._retriever.retrieve_relevant_content(query=query, seed_entities=entities,
                                                                   type=Retriever.CHUNK, mode="aug_ppr")
            return retrieved_passages

    async def generation_qa(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        # For FastGraphRAG 
        if self.config.augmentation_ppr:
            # 截断context以符合token限制
            context = self._truncate_context(context)
            msg = QueryPrompt.GENERATE_RESPONSE_QUERY_WITH_REFERENCE.format(query=query, context=context)
            try:
                return await self.llm.aask(msg=msg)
            except Exception as e:
                err = str(e)
                if 'data_inspection_failed' in err or 'inappropriate content' in err:
                    logger.warning(f"PPRQuery: augmentation_ppr 内容审查失败，返回空: {err}")
                    return ""
                raise
        else:
            # For HippoRAG. Note that 5 is the default number of passages to retrieve for HippoRAG souce code
            # Please refer to: https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/ircot_hipporag.py#L289 
            retrieved_passages = context[:self.config.num_doc]

            working_memory = Memory()
            instruction = QueryPrompt.COT_SYSTEM_DOC if len(retrieved_passages) else QueryPrompt.COT_SYSTEM_NO_DOC
            working_memory.add(Message(content=instruction, role='system'))
            
            user_prompt = ''
            for passage in retrieved_passages:
                # 处理TextChunk对象，提取content内容
                if hasattr(passage, 'content'):
                    user_prompt += f' {passage.content}\n\n'
                else:
                    # 兼容字符串格式
                    user_prompt += f' {passage}\n\n'
            user_prompt += 'Question: ' + query + '\nThought: '
            
            # 截断user_prompt以符合token限制
            user_prompt = self._truncate_context(user_prompt)
            working_memory.add(Message(content=user_prompt, role='user'))

            # 将working_memory转换为标准消息格式
            messages = [msg.to_dict() for msg in working_memory.get()]

            try:
                # 使用working_memory管理对话历史，传递空的system_msgs避免冲突
                try:
                    response = await self.llm.aask(msg=messages, system_msgs=[])
                except Exception as e:
                    err = str(e)
                    if 'data_inspection_failed' in err or 'inappropriate content' in err:
                        print('QA read exception', e)
                        return ''
                    raise
            except Exception as e:
                print('QA read exception', e)
                return ''
        return response
    
    async def generation_summary(self, query, context):
        pass