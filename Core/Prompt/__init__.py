"""
Core.Prompt - 提示模板模块
包含所有用于不同查询方法的提示模板
"""

from .QueryPrompt import *
from .GraphPrompt import *
from .EntityPrompt import *
from .CommunityPrompt import *
from .RaptorPrompt import *
from .TogPrompt import *
from .HaoPrompt import *

__all__ = [
    # QueryPrompt
    "GLOBAL_MAP_RAG_POINTS",
    "FAIL_RESPONSE", 
    "GLOBAL_REDUCE_RAG_RESPONSE",
    "LOCAL_RAG_RESPONSE",
    "KEYWORDS_EXTRACTION",
    "RAG_RESPONSE",
    "IRCOT_REASON_INSTRUCTION",
    "KGP_REASON_PROMPT",
    "KGP_QUERY_PROMPT",
    "GENERATE_RESPONSE_QUERY_WITH_REFERENCE",
    "COT_SYSTEM_DOC",
    "COT_SYSTEM_NO_DOC",
    "DALK_RERANK_PROMPT",
    "DALK_CONVERT_PROMPT", 
    "DALK_STEP_PROMPT",
    "DALK_CONTEXT_PROMPT",
    "DALK_QUERY_PROMPT",
    
    # HaoPrompt
    "REASONING_STEP_PROMPT",
    "FORCE_ANSWER_PROMPT",
    "REASONING_STEP_PROMPT_QWEN",
    "FORCE_ANSWER_PROMPT_QWEN",
    "NO_CONTEXT_AVAILABLE",
] 