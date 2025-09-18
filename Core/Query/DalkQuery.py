from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from Core.Prompt.QueryPrompt import DALK_RERANK_PROMPT, DALK_CONVERT_PROMPT, DALK_STEP_PROMPT, DALK_CONTEXT_PROMPT, DALK_QUERY_PROMPT

class DalkQuery(BaseQuery):
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
        logger.info(f"DalkQuery上下文token限制: {self.max_context_tokens} (模型MAX_MODEL_LEN: {model_max_len}, max_token: {max_token})")

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
            logger.info(f"DalkQuery上下文截断: {original_tokens} tokens -> {truncated_tokens} tokens，耗时: {elapsed_time:.3f}s")
            
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

    async def _select_most_relative_paths(self, path_list, query):
        if path_list is None: return ""
        # import pdb
        # pdb.set_trace()
        path_str = [(path[0]["src_id"]
                     + ("->" + edge["content"] + "-> " + edge["tgt_id"]) for edge in path) # haloyang 完善获取content代码
                     for path in path_list]
        context = DALK_RERANK_PROMPT.format(graph=path_str, question=query)

        try:
            return await self.llm.aask(context)
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"DalkQuery: 路径重排触发内容审查，跳过: {err}")
                return ""
            raise

    async def _select_most_relative_neis(self, nei_list, query):
        # import pdb
        # pdb.set_trace()
        if nei_list is None: return ""
        nei_str_list = ["->".join([e["src_id"], e["content"], e["tgt_id"]])
                        for e in nei_list]
        if len(nei_str_list) > 5:
            nei_str_list = nei_str_list[:-5]
        neighbor_str = "\n".join(nei_str_list)
        context = DALK_RERANK_PROMPT.format(graph=neighbor_str, question=query)

        try:
            return await self.llm.aask(context)
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"DalkQuery: 邻居重排触发内容审查，跳过: {err}")
                return ""
            raise

    async def _to_natural_language(self, context):
        context = DALK_CONVERT_PROMPT.format(graph=context)
        try:
            return await self.llm.aask(context)
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"DalkQuery: 自然语言转换触发内容审查，跳过: {err}")
                return ""
            raise

    async def _process_info(self, query, paths, neis):
        paths = await self._to_natural_language(paths)
        neis = await self._to_natural_language(neis)
        try:
            step = await self.llm.aask(DALK_STEP_PROMPT.format(question=query, paths=paths, neis=neis))
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"DalkQuery: 生成step触发内容审查，跳过: {err}")
                step = ""
            else:
                raise
        return paths, neis, step

    async def _retrieve_relevant_contexts(self, query):
        entities = await self.extract_query_entities(query)
        entities = await self._retriever.retrieve_relevant_content(type = Retriever.ENTITY, mode = "link_entity", query_entities = entities)
        entities = [node['entity_name'] for node in entities]
        paths = await self._select_most_relative_paths(
                    await self._retriever.retrieve_relevant_content(type = Retriever.SUBGRAPH, mode = "paths_return_list", seed = entities, cutoff = self.config.k_hop),
                    query
                    )
        neis = await self._select_most_relative_neis(
                    await self._retriever.retrieve_relevant_content(type = Retriever.SUBGRAPH, mode = "neighbors_return_list", seed = entities),
                    query
                    )
        paths, neis, step = await self._process_info(query, paths, neis)
        return DALK_CONTEXT_PROMPT.format(paths=paths, neis=neis, step=step)

    async def query(self, query):
        context = await self._retrieve_relevant_contexts(query)
        try:
            response = await self.llm.aask(DALK_QUERY_PROMPT.format(question=query, context=context))
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"DalkQuery: 最终问答触发内容审查，降级为空: {err}")
                response = ""
            else:
                raise
        logger.info(query)
        logger.info(response)
        return {
            "response": response,
            "context": context
        }

    async def generation_qa(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE
        context_str = '\n'.join('{index}: {context}'.format(index=i, context=c) for i, c in enumerate(context, start=1))
        
        # 截断context以符合token限制
        context_str = self._truncate_context(context_str)

        try:
            answer = await self.llm.aask(DALK_QUERY_PROMPT.format(question=query, context=context_str))
            return answer
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"DalkQuery: generation_qa 内容审查，返回占位: {err}")
                return "由于内容审查限制，已跳过该条请求。"
            raise

    async def generation_summary(self, query, context):

        if context is None:
            return QueryPrompt.FAIL_RESPONSE
