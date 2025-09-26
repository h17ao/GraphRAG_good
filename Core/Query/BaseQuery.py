import asyncio
from abc import ABC, abstractmethod
from Core.Retriever.MixRetriever import MixRetriever
from typing import Any
from Core.Prompt import GraphPrompt, QueryPrompt
from Core.Common.Utils import clean_str, prase_json_from_response, truncate_list_by_token_size, \
    list_to_quoted_csv_string
from Core.Common.Logger import logger
from Core.Common.LLM import RetrievalLLM
from Core.Common.Context import Context


class BaseQuery(ABC):
    def __init__(self, config, retriever_context):
        self._retriever = MixRetriever(retriever_context)
        self.config = config
        
        # 从retriever_context.context字典获取完整的Config对象来初始化检索专用LLM
        # retriever_context.context["full_config"]包含完整的Config对象
        full_config = retriever_context.context.get("full_config")
        if not full_config:
            raise ValueError("无法从retriever_context获取完整配置，检索LLM初始化失败")
        
        # 使用完整的Config对象初始化Context
        context = Context()
        context.config = full_config
        self.llm = RetrievalLLM(context)
        logger.info(f"检索阶段使用模型: {self.llm.config.model}")

    @abstractmethod
    async def _retrieve_relevant_contexts(self, **kwargs):
        pass

    #   修改为同时保存context且实时读写保存    
    async def query(self, query):
        context = await self._retrieve_relevant_contexts(query=query)
        response = None
        if self.config.query_type == "summary":
            response = await self.generation_summary(query, context)
        elif self.config.query_type == "qa":
            response = await self.generation_qa(query, context)
        else:
            logger.error("Invalid query type")
        
        # 默认返回包含context的完整结果
        return {
            "response": response,
            "context": context
        }

    @abstractmethod
    async def generation_summary(self, query, context):
        pass

    @abstractmethod
    async def generation_qa(self, query, context):
        pass

    async def extract_query_entities(self, query):
        entities = []
        try:
            ner_messages = GraphPrompt.NER.format(user_input=query)
            try:
                response_content = await self.llm.aask(ner_messages)
            except Exception as e:
                err = str(e)
                if 'data_inspection_failed' in err or 'inappropriate content' in err:
                    logger.warning(f"BaseQuery: NER 调用内容审查失败，降级为空: {err}")
                    return []
                raise

            entities = prase_json_from_response(response_content)
            if 'named_entities' not in entities:
                entities = []
            else:
                entities = entities['named_entities']
            entities = [clean_str(p) for p in entities]
        except Exception as e:
            logger.error('Error in Retrieval NER: {}'.format(e))

        return entities

    async def extract_query_keywords(self, query, mode="low"):
        kw_prompt = QueryPrompt.KEYWORDS_EXTRACTION.format(query=query)
        try:
            result = await self.llm.aask(kw_prompt)
        except Exception as e:
            err = str(e)
            if 'data_inspection_failed' in err or 'inappropriate content' in err:
                logger.warning(f"BaseQuery: 关键词抽取内容审查失败，降级为空: {err}")
                return ""
            raise
        keywords = None
        keywords_data = prase_json_from_response(result)
        if mode == "low":
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        elif mode == "high":
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)
        elif mode == "hybrid":
            low_level = keywords_data.get("low_level_keywords", [])
            high_level = keywords_data.get("high_level_keywords", [])
            keywords = [low_level, high_level]

        return keywords

    async def _map_global_communities(
            self,
            query: str,
            communities_data
    ):

        # TODO: support other type of context filter
        community_groups = []
        while len(communities_data):
            this_group = truncate_list_by_token_size(
                communities_data,
                key=lambda x: x["report_string"],
                max_token_size=self.config.global_max_token_for_community_report,
            )
            community_groups.append(this_group)
            communities_data = communities_data[len(this_group):]

        async def _process(community_truncated_datas: list[Any]) -> dict:
            communities_section_list = [["id", "content", "rating", "importance"]]
            for i, c in enumerate(community_truncated_datas):
                communities_section_list.append(
                    [
                        i,
                        c["report_string"],
                        c["report_json"].get("rating", 0),
                        c['occurrence'], #   删去['community_info']['occurrence']，纠正了错误的字段访问路径
                    ]
                )
            community_context = list_to_quoted_csv_string(communities_section_list)
            sys_prompt_temp = QueryPrompt.GLOBAL_MAP_RAG_POINTS
            sys_prompt = sys_prompt_temp.format(context_data=community_context)

            response = await self.llm.aask(
                query,
                system_msgs=[sys_prompt]
            )

            data = prase_json_from_response(response)
            return data.get("points", [])

        logger.info(f"Grouping to {len(community_groups)} groups for global search")
        responses = await asyncio.gather(*[_process(c) for c in community_groups])

        return responses
