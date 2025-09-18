from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt


class KGPQuery(BaseQuery):
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
        logger.info(f"KGPQuery上下文token限制: {self.max_context_tokens} (模型MAX_MODEL_LEN: {model_max_len}, max_token: {max_token})")

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
            logger.info(f"KGPQuery上下文截断: {original_tokens} tokens -> {truncated_tokens} tokens，耗时: {elapsed_time:.3f}s")
            
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

    async def _retrieve_relevant_contexts(self, query):
        corpus, candidates_idx = await self._retriever.retrieve_relevant_content(key="description",
                                                                                 type=Retriever.ENTITY, mode="all")
        cur_contexts, idxs = await self._retriever.retrieve_relevant_content(seed=query, corpus=corpus,
                                                                             candidates_idx=candidates_idx,
                                                                             top_k=self.config.top_k // self.config.nei_k,
                                                                             type=Retriever.ENTITY, mode="tf_df")
        contexts = []
        next_reasons = []
        for context in cur_contexts:
            try:
                reason = await self.llm.aask(QueryPrompt.KGP_REASON_PROMPT.format(question=query, context=context))
                next_reasons.append(query + '\n' + reason)
            except Exception as e:
                error_msg = str(e)
                if 'data_inspection_failed' in error_msg or 'inappropriate content' in error_msg:
                    logger.warning(f"跳过内容审查失败的推理生成: {error_msg}")
                    # 跳过这个context，使用原始query作为reason
                    next_reasons.append(query + '\n' + "内容审查跳过")
                else:
                    # 其他类型的错误仍然抛出
                    raise e

        logger.info("next_reasons: {next_reasons}".format(next_reasons=next_reasons))

        visited = []

        for idx, next_reason in zip(idxs, next_reasons):
            nei_candidates_idx = await self._retriever.retrieve_relevant_content(seed=idx, type=Retriever.ENTITY,
                                                                                 mode="by_neighbors")
            nei_candidates_idx = [_ for _ in nei_candidates_idx if _ not in visited]
            if nei_candidates_idx == []:
                continue

            next_contexts, next_idxs = await self._retriever.retrieve_relevant_content(seed=next_reason, corpus=corpus,
                                                                                       candidates_idx=nei_candidates_idx,
                                                                                       top_k=self.config.nei_k,
                                                                                       type=Retriever.ENTITY,
                                                                                       mode="tf_df")
            contexts.extend([corpus[idx] + '\n' + corpus[_] for _ in next_idxs if corpus[_] != corpus[idx]])
            visited.append(idx)
            visited.extend([_ for _ in next_idxs])
        return contexts

    async def generation_qa(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        context_str = '\n'.join('{index}: {context}'.format(index=i, context=c) for i, c in enumerate(context, start=1))
        
        # 截断context以符合token限制
        context_str = self._truncate_context(context_str)

        try:
            answer = await self.llm.aask(QueryPrompt.KGP_QUERY_PROMPT.format(question=query, context=context_str))
            return answer
        except Exception as e:
            error_msg = str(e)
            if 'data_inspection_failed' in error_msg or 'inappropriate content' in error_msg:
                logger.warning(f"跳过内容审查失败的问答生成: {error_msg}")
                return "抱歉，由于内容审查限制，无法生成回答。请尝试其他查询。"
            else:
                # 其他类型的错误仍然抛出
                raise e

    async def generation_summary(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE
