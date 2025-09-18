"""
HaoPrompt - HaoQuery使用的提示模板
从HaoQuery.py中提取的原有提示模板
"""

REASONING_STEP_PROMPT_QWEN = """
**User query**: {query}

------------------------------------------------------------------------------------------------

**Graph evidence**: {context_text}

------------------------------------------------------------------------------------------------

**TASK**: You are a reasoning assistant working with information from a knowledge graph.

Your task is to answer the user query with the provided graph evidence and your own knowledge.
Follow this process for each query:

1. Carefully read the user query and the retrieved graph evidence.
2. Think step-by-step about what the query is asking and what the evidence supports.
3. Use both the graph evidence and your own knowledge to reason through the query.

- If the evidence **is sufficient**, respond in the following format:
  <label> able </label>  
  <answer> final answer here. </answer>

- If the evidence **is not sufficient**, do the following:
  - Identify what key information is missing to answer the query.
  - Propose a natural, specific follow-up question that would help obtain the missing evidence.
  - Respond in the following format:   
  <label> unable </label>  
  <query> proposed follow-up question here. </query>

Do not include any text outside the specified tags. Maintain strict adherence to the format."""

# 强制作答提示模板 - 从HaoQuery._force_answer中提取
FORCE_ANSWER_PROMPT_QWEN = """
**User query**: {query}

------------------------------------------------------------------------------------------------

**Graph evidence**: {context_text}

------------------------------------------------------------------------------------------------

**TASK**: You are a reasoning assistant working with information from a knowledge graph.

Your task is to answer the user query with the provided graph evidence and your own knowledge.
Follow this process for each query:

1. Carefully read the user query and the retrieved graph evidence.
2. Think step-by-step about what the query is asking and what the evidence supports.
3. Use both the graph evidence and your own knowledge to reason through the query.

- then respond in the following format: 
  <answer> final answer here. </answer>
  
Do not include any text outside the specified tags. Maintain strict adherence to the format. """


REASONING_STEP_PROMPT = """
**User query**: {query}

------------------------------------------------------------------------------------------------

**Graph evidence**: {context_text}

------------------------------------------------------------------------------------------------

**TASK**: You are a reasoning assistant working with information from a knowledge graph.

Your task is to answer the user query with the provided graph evidence and your own knowledge.
Follow this process for each query:

1. Carefully read the user query and the retrieved graph evidence.
2. Think step-by-step about what the query is asking and what the evidence supports.
3. Use both the graph evidence and your own knowledge to reason through the query.

- If the evidence **is sufficient**, respond in the following format:  
  <think> reasoning process here. </think>  
  <label> able </label>  
  <answer> final answer here. </answer>

- If the evidence **is not sufficient**, do the following:
  - Identify what key information is missing to answer the query.
  - Propose a natural, specific follow-up question that would help obtain the missing evidence.
  - Respond in the following format: 
  <think> reasoning process here. </think>  
  <label> unable </label>  
  <query> proposed follow-up question here. </query>

Do not include any text outside the specified tags. Maintain strict adherence to the format."""

# 强制作答提示模板 - 从HaoQuery._force_answer中提取
FORCE_ANSWER_PROMPT = """
**User query**: {query}

------------------------------------------------------------------------------------------------

**Graph evidence**: {context_text}

------------------------------------------------------------------------------------------------

**TASK**：You are a reasoning assistant working with information from a knowledge graph.

Your task is to answer the user query with the provided graph evidence and your own knowledge.
Follow this process for each query:

1. Carefully read the user query and the retrieved graph evidence.
2. Think step-by-step about what the query is asking and what the evidence supports.
3. Use both the graph evidence and your own knowledge to reason through the query.

- then respond in the following format: 
  <think> reasoning process here. </think>  
  <answer> final answer here. </answer>

Do not include any text outside the specified tags. Maintain strict adherence to the format."""


# 实体关系抽取提示模板 - 参考KG_AGNET一步法
ENTITY_RELATION_EXTRACTION_PROMPT = """You are tasked with extracting nodes and relationships from given content. Here's what you need to do:

Content Extraction:
You should process input content and identify entities mentioned within it. Entities can be any noun phrases or concepts that represent distinct entities in the context of the given content.

Node Extraction:
Each identified entity should have a unique numeric identifier (id).

Relationship Extraction:
You should identify relationships between entities mentioned in the content.
A Relationship should have a source (src) and a destination (dst) which are node IDs representing the entities involved in the relationship.

Instructions for you:
Provide the extracted nodes and relationships in the specified JSON format below.
Note that you must not output any code or additional explanations! Just output the JSON.

Example Content:
Input:
John works at XYZ Corporation. The company is located in New York City.

Expected Output:
{{
  "nodes": [
    {{"id": 1, "entity": "John"}},
    {{"id": 2, "entity": "XYZ Corporation"}},
    {{"id": 3, "entity": "New York City"}}
  ],
  "edges": [
    {{"src": 1, "relation": "works at", "dst": 2}},
    {{"src": 2, "relation": "located in", "dst": 3}}
  ]
}}

===== TASK =====
Input:
{text_content}"""

# 常量定义
NO_CONTEXT_AVAILABLE = "No context available" 