from Core.Query.BaseQuery import BaseQuery
from Core.Common.Logger import logger
from Core.Common.Constants import Retriever
from Core.Common.Utils import list_to_quoted_csv_string, truncate_list_by_token_size, combine_contexts
from Core.Prompt import QueryPrompt
from typing import Union


class ToGQuery(BaseQuery):
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
        logger.info(f"ToGQuery上下文token限制: {self.max_context_tokens} (模型MAX_MODEL_LEN: {model_max_len}, max_token: {max_token})")

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
            logger.info(f"ToGQuery上下文截断: {original_tokens} tokens -> {truncated_tokens} tokens，耗时: {elapsed_time:.3f}s")
            
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
    
    def _update_pre_heads_length(self):
        """确保pre_heads长度与topic_entity_candidates一致"""
        try:
            # 检查topic_entity_candidates是否为空或None
            if not hasattr(self, 'topic_entity_candidates') or not self.topic_entity_candidates:
                logger.warning("topic_entity_candidates is empty or None, initializing pre_heads as empty list")
                self.pre_heads = []
                return
            
            # 检查pre_heads是否已初始化
            if not hasattr(self, 'pre_heads') or self.pre_heads is None:
                logger.warning("pre_heads is not initialized, creating new list")
                self.pre_heads = [-1] * len(self.topic_entity_candidates)
                return
                
            current_length = len(self.topic_entity_candidates)
            pre_heads_length = len(self.pre_heads)
            
            logger.debug(f"Updating pre_heads length: candidates={current_length}, pre_heads={pre_heads_length}")
            
            if current_length > pre_heads_length:
                # 如果candidates变多了，用-1填充pre_heads
                self.pre_heads.extend([-1] * (current_length - pre_heads_length))
                logger.debug(f"Extended pre_heads to length {len(self.pre_heads)}")
            elif current_length < pre_heads_length:
                # 如果candidates变少了，截断pre_heads
                self.pre_heads = self.pre_heads[:current_length]
                logger.debug(f"Truncated pre_heads to length {len(self.pre_heads)}")
        except Exception as e:
            logger.error(f"Error in _update_pre_heads_length: {e}")
            # 安全的默认处理
            if hasattr(self, 'topic_entity_candidates') and self.topic_entity_candidates:
                self.pre_heads = [-1] * len(self.topic_entity_candidates)
            else:
                self.pre_heads = []

    async def _extend_reasoning_paths_per_depth(self, query: str) -> tuple[bool, list[list[tuple]], list, list, list]:
        from collections import defaultdict
        total_entity_relation_list = []
        total_relations_dict = defaultdict(list)
        
        # 确保pre_heads长度与topic_entity_candidates一致
        self._update_pre_heads_length()

        ####################### Retrieve relations from entities by agent ############################################
        # 额外的安全检查
        if not self.topic_entity_candidates:
            logger.warning("topic_entity_candidates is empty, returning empty results")
            return False, [], [], [], []
        
        for index, entity in enumerate(self.topic_entity_candidates):
            if entity != "[FINISH]":
                # 多重安全检查pre_heads访问
                try:
                    if hasattr(self, 'pre_heads') and self.pre_heads and index < len(self.pre_heads):
                        pre_head = self.pre_heads[index]
                    else:
                        pre_head = -1
                        logger.debug(f"Using default pre_head=-1 for index {index}")
                except (IndexError, AttributeError, TypeError) as e:
                    logger.warning(f"Error accessing pre_heads[{index}]: {e}, using default value -1")
                    pre_head = -1
                
                relations, relations_dict = await self._retriever.retrieve_relevant_content(type=Retriever.RELATION, # aim at retrieving relations
                                                                                            mode="from_entity_by_agent", # method: use information of entity by agent
                                                                                            query=query,
                                                                                            entity=entity,
                                                                                            pre_relations_name=self.pre_relations_name,
                                                                                            pre_head=pre_head,
                                                                                            width=self.config.width)
                # merge
                for key, value in relations_dict.items():
                    total_relations_dict[key].extend(value)
                total_entity_relation_list.extend(relations)

        ###################### Retrieve entities from relations by agent #############################################
        flag, candidates_list = await self._retriever.retrieve_relevant_content(
            type=Retriever.ENTITY, # aim at retrieving entity
            mode="from_relation_by_agent", # method: use information of relation by agent
            query=query,
            total_entity_relation_list=total_entity_relation_list,
            total_relations_dict=total_relations_dict,
            width=self.config.width,
        )

        ####################### Assemble relations and entities to obtain paths ########################################
        # Check if candidates_list is empty to prevent unpacking error
        if not candidates_list:
            logger.warning(f"candidates_list is empty for query: {query}. Returning empty results.")
            return False, [], [], [], []
        
        try:
            # 安全地解压缩candidates_list
            rels, candidates, tops, heads, scores = map(list, zip(*candidates_list))
            
            # 确保所有列表长度一致
            min_length = min(len(tops), len(rels), len(candidates))
            if min_length != len(candidates):
                logger.warning(f"Length mismatch in unpacked data: tops={len(tops)}, rels={len(rels)}, candidates={len(candidates)}")
                tops = tops[:min_length]
                rels = rels[:min_length]
                candidates = candidates[:min_length]
                heads = heads[:min_length]
            
            # 安全构建chains
            chains = []
            if min_length > 0:
                chains = [[(tops[i], rels[i], candidates[i]) for i in range(min_length)]]
            
            logger.debug(f"Successfully assembled paths: {len(chains)} chains with {min_length} elements")
            return flag, chains, candidates, rels, heads
            
        except Exception as e:
            logger.error(f"Error assembling reasoning paths: {e}")
            return False, [], [], [], []

    async def _retrieve_initialization(self, query: str):
        # find relevant entity seeds
        entities = await self.extract_query_entities(query)
        # link these entity seeds to our actual entities in vector database and our built graph
        entities = await self._retriever.retrieve_relevant_content(type=Retriever.ENTITY,
                                                             mode="link_entity",
                                                             query_entities=entities)
        entities = list(map(lambda x: x['entity_name'], entities))
        # define the global variable about reasoning
        self.reasoning_paths_list = []
        self.pre_relations_name = []
        self.pre_heads = [-1] * len(entities)
        self.flag_printed = False
        self.topic_entity_candidates = entities

    async def _retrieve_relevant_contexts(self, query: str, mode: str) -> str:
        if mode == "retrieve":
            flag, chains, candidates, rels, heads = await self._extend_reasoning_paths_per_depth(query)
            # update
            self.pre_relations_name = rels
            self.pre_heads = heads
            self.reasoning_paths_list.append(chains)
            self.topic_entity_candidates = candidates
            # 确保pre_heads长度与topic_entity_candidates一致
            self._update_pre_heads_length()
            # If we can retrieve some paths, then try to response the query using current information.
            if flag:
                from Core.Prompt.TogPrompt import prompt_evaluate
                context = prompt_evaluate + query + "\n"
                chain_prompt = '\n'.join(
                    [', '.join([str(x) for x in chain]) for sublist in self.reasoning_paths_list for chain in sublist])
                context += "\nKnowledge Triplets: " + chain_prompt + 'A: '
                return context
            else:
                return ""

        elif mode == "half_stop":
            from Core.Prompt.TogPrompt import answer_prompt
            context = answer_prompt.format(query + '\n')
            chain_prompt = '\n'.join(
                [', '.join([str(x) for x in chain]) for sublist in self.reasoning_paths_list for chain in sublist])
            context += "\nKnowledge Triplets: " + chain_prompt + 'A: '

            return context
        else:
            raise TypeError("parameter mode must be either retrieve or half_stop!")

    def _encapsulate_answer(self, query, answer, chains):
        dic = {"question": query, "results": answer, "reasoning_chains": chains}
        return dic

    def _is_finish_list(self)->tuple[bool, list]:
        if all(elem == "[FINISH]" for elem in self.topic_entity_candidates):
            return True, []
        else:
            new_lst = [elem for elem in self.topic_entity_candidates if elem != "[FINISH]"]
            return False, new_lst

    async def _half_stop(self, query, depth):
        logger.info("No new knowledge added during search depth %d, stop searching." % depth)
        context = await self._retrieve_relevant_contexts(query=query, mode="half_stop")
        response = await self.generation_qa(mode="half_stop", context=context)
        return response

    async def query(self, query):
        # Initialization and find the entitiy seeds about given query
        await self._retrieve_initialization(query)
        all_contexts = []  # 用于收集所有上下文
        
        for depth in range(1, self.config.depth + 1):
            context = await self._retrieve_relevant_contexts(query=query, mode="retrieve")
            if context:
                all_contexts.append(f"Depth {depth}: {context}")
                # If the length of context is not zero, we try to answer the question
                is_stop, response = await self.generation_qa(mode="retrieve", context=context)
                if is_stop:
                    # If is_stop is True, we can answer the question without searching further
                    logger.info("ToG stopped at depth %d." % depth)
                    break
                else:
                    # Otherwise, we still need to search
                    logger.info("ToG still not find the answer at depth %d." % depth)
                    all_candidates_are_finish, new_candidates = self._is_finish_list()
                    if all_candidates_are_finish:
                        # If all the item in candidates is [FINISH], then stop
                        response = await self._half_stop(query, depth)
                        break
                    else:
                        # Otherwise, continue searching
                        self.topic_entity_candidates = new_candidates
                        # 确保pre_heads长度与topic_entity_candidates一致
                        self._update_pre_heads_length()
            else:
                # Otherwise, it means no more new knowledge paths can be found.
                response = await self._half_stop(query, depth)
                break
        # This is the evidence chain.
        #print(self.reasoning_paths_list)
        
        # 返回与其他查询方法相同的字典格式
        return {
            "response": response,
            "context": "\n".join(all_contexts) if all_contexts else ""
        }

    async def generation_qa(self, mode: str, context: str) -> Union[tuple[bool, str], str]:
        # 截断context以符合token限制
        context = self._truncate_context(context)
        
        messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
                    {"role": "user", "content": context}]
        try:
            response = await self.llm.aask(msg=messages)
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"ToGQuery: 内容审查失败，返回占位: {err}")
                response = "由于内容审查限制，已跳过该条请求。"
            else:
                raise

        if mode == "retrieve":
            # extract is_stop answer from response
            start_index = response.find("{")
            end_index = response.find("}")
            if start_index != -1 and end_index != -1:
                is_stop = response[start_index + 1:end_index].strip()
            else:
                is_stop = ""

            # determine whether it can stop.
            is_true = lambda text: text.lower().strip().replace(" ", "") == "yes"
            if is_true(is_stop):
                return True, response
            else:
                return False, response
        elif mode == "half_stop":
            return response
        else:
            raise TypeError("parameter mode must be either retrieve or half_stop!")

    async def generation_summary(self, query, context):
        if context is None:
            return QueryPrompt.FAIL_RESPONSE