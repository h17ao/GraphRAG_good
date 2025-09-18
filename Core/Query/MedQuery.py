from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Prompt import QueryPrompt
from typing import Union
import asyncio
# from torch_geometric.data import Data, InMemoryDataset
from typing import Any, Dict, List, Tuple, no_type_check
from pcst_fast import pcst_fast
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from Core.Common.Utils import truncate_str_by_token_size
from Core.Common.Constants import GRAPH_FIELD_SEP

class MedQuery(BaseQuery):
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
        logger.info(f"MedQuery上下文token限制: {self.max_context_tokens} (模型MAX_MODEL_LEN: {model_max_len}, max_token: {max_token})")

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
            logger.info(f"MedQuery上下文截断: {original_tokens} tokens -> {truncated_tokens} tokens，耗时: {elapsed_time:.3f}s")
            
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

    async def _concatenate_information(self, metagraph_relation: str, metagraph_edge: tuple[str, str]):
        metagraph_relation_seperated = metagraph_relation.split(GRAPH_FIELD_SEP)
        return list(map(lambda x: metagraph_edge[0] + " " + x + " " +metagraph_edge[1], metagraph_relation_seperated))


    async def _retrieve_relevant_contexts(self, query: str):
        # find the most relevant subgraph
        context = await self._retriever.retrieve_relevant_content(type=Retriever.SUBGRAPH, mode="concatenate_information_return_list", seed=query) # return list
        if context is None or len(context) == 0:
            logger.warning(f"No subgraph context found for query: {query}")
            return None
        chunk_id = context[0]["source_id"]
        metagraph_nodes = set()
        iteration_count = 1
        while len(metagraph_nodes) < self.config.topk_entity:
            origin_nodes = await self._retriever.retrieve_relevant_content(seed=query,
                                                                           top_k=self.config.topk_entity * iteration_count,
                                                                           type=Retriever.ENTITY,
                                                                           mode="vdb")  # list[dict]
            for node in origin_nodes[(iteration_count - 1) * self.config.topk_entity:]:
                if node["source_id"] == chunk_id and node["entity_name"] not in metagraph_nodes:
                    metagraph_nodes.add(node["entity_name"])

            iteration_count += 1

        logger.info("Find top {} entities at iteration {}!".format(self.config.topk_entity, iteration_count))

        metagraph_nodes = await self._retriever.retrieve_relevant_content(seed=list(metagraph_nodes), k=self.config.k_hop, type=Retriever.SUBGRAPH, mode="k_hop_return_set") # return set
        metagraph = await self._retriever.retrieve_relevant_content(seed=list(metagraph_nodes), type=Retriever.SUBGRAPH, mode="induced_subgraph_return_networkx")  # return networkx
        metagraph_edges = list(metagraph.edges())
        metagraph_super_relations = await self._retriever.retrieve_relevant_content(seed=metagraph_edges, type=Retriever.RELATION, mode="by_source&target") # super relations
        zip_combined = tuple(zip(metagraph_super_relations, metagraph_edges))
        metagraph_concatenate = await asyncio.gather(*[self._concatenate_information(metagraph_super_relation, metagraph_edge) for metagraph_super_relation, metagraph_edge in zip_combined]) # list[list]

        context = ""
        for relations in metagraph_concatenate:
            context += ", ".join(relations)
            context += ", "
        return context



    async def query(self, query):
        context = await self._retrieve_relevant_contexts(query)
        print(context)
        if context is None:
            logger.warning(f"No context found for query: {query}")
            return QueryPrompt.FAIL_RESPONSE
        response = await self.generation_qa(query, context)
        return response

    async def generation_qa(self, query: str, context: str):
        # 截断context以符合token限制
        context = self._truncate_context(context)
        
        messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": "the question is: " + query + ", the provided information is (list seperated by ,): " +  context}]
        try:
            response = await self.llm.aask(msg=messages)
            return response
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"MedQuery: 内容审查失败，跳过: {err}")
                return "由于内容审查限制，已跳过该条请求。"
            raise

    async def generation_summary(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE