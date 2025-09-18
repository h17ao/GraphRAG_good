#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
from dataclasses import field
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pydantic import BaseModel
from Config import *
from Core.Common.Constants import CONFIG_ROOT, GRAPHRAG_ROOT
from Core.Utils.YamlModel import YamlModel


class WorkingParams(BaseModel):
    """Working parameters"""

    working_dir: str = ""
    exp_name: str = ""
    data_root: str = ""
    dataset_name: str = ""
    method_name: str = ""


class Config(WorkingParams, YamlModel):
    """Configurations for our project"""

    # Key Parameters
    llm: LLMConfig
    retrieval_llm: Optional[LLMConfig] = None  # 检索专用LLM配置
    eval_llm: Optional[LLMConfig] = None       # 评估专用LLM配置
    exp_name: str = "default"
    # RAG Embedding
    embedding: EmbeddingConfig = EmbeddingConfig()

    # Basic Config
    use_entities_vdb: bool = True
    use_relations_vdb: bool = True  # Only set True for LightRAG
    use_subgraphs_vdb: bool = False  # Only set True for Medical-GraphRAG
    vdb_type: str = "vector"  # vector/colbert
    token_model: str = "gpt-3.5-turbo"
    
    
    # Chunking
    chunk: ChunkConfig = ChunkConfig()
    # chunk_token_size: int = 1200
    # chunk_overlap_token_size: int = 100
    # chunk_method: str = "chunking_by_token_size"
   
    llm_model_max_token_size: int = 32768
  
    use_entity_link_chunk: bool = True  # Only set True for HippoRAG and FastGraphRAG
    
    # Graph Config
    graph: GraphConfig = GraphConfig()
    
    # Retrieval Parameters
    retriever: RetrieverConfig = RetrieverConfig()


    # ColBert Option
    use_colbert: bool = True
    colbert_checkpoint_path: str = "/home/yingli/HippoRAG/exp/colbertv2.0"
    index_name: str = "nbits_2"
    similarity_max: float = 1.0
    # Graph Augmentation
    enable_graph_augmentation: bool = True

    # Query Config 
    query: QueryConfig = QueryConfig()
  
    @classmethod
    def from_yaml_config(cls, path: str):
        """user config llm
        example:
        llm_config = {"api_type": "xxx", "api_key": "xxx", "model": "xxx"}
        gpt4 = Option.from_llm_config(llm_config)
        A = Role(name="A", profile="Democratic candidate", goal="Win the election", actions=[a1], watch=[a2], config=gpt4)
        """
        opt = parse(path)
        return Config(**opt)

    @classmethod
    def parse(cls, _path, dataset_name, method_name=None):
        """Parse config from yaml file"""
        opt = [parse(_path)]

        default_config_paths: List[Path] = [
            GRAPHRAG_ROOT / "Option/Config2.yaml",
            CONFIG_ROOT / "Config2.yaml",
        ]
        opt += [Config.read_yaml(path) for path in default_config_paths]
    
        final = merge_dict(opt)
        final["dataset_name"] = dataset_name
        final["method_name"] = method_name or ""
        
        # 如果环境变量中有EXP_NAME，优先使用环境变量的值
        import os
        env_exp_name = os.environ.get("EXP_NAME")
        if env_exp_name:
            final["exp_name"] = env_exp_name
        
        # 构建working_dir: dataset_name/method_name (如果有method_name)
        if method_name:
            final["working_dir"] = os.path.join(final["working_dir"], dataset_name, method_name)
        else:
            final["working_dir"] = os.path.join(final["working_dir"], dataset_name)
        
        # 创建Config实例
        config_instance = Config(**final)
        
        # 重新配置logger以使用正确的路径
        from Core.Common.Logger import reconfigure_logger_for_method_dataset
        # 获取原始working_dir（不含dataset_name/method_name）
        original_working_dir = final.get("working_dir", "./")
        if method_name:
            # 移除 dataset_name/method_name 部分
            suffix_to_remove = f"/{dataset_name}/{method_name}"
            if original_working_dir.endswith(suffix_to_remove):
                original_working_dir = original_working_dir[:-len(suffix_to_remove)]
            elif original_working_dir.endswith(suffix_to_remove.replace("/", "\\")):
                original_working_dir = original_working_dir[:-len(suffix_to_remove.replace("/", "\\"))]
        else:
            # 移除 dataset_name 部分
            if original_working_dir.endswith(f"/{dataset_name}"):
                original_working_dir = original_working_dir[:-len(f"/{dataset_name}")]
            elif original_working_dir.endswith(f"\\{dataset_name}"):
                original_working_dir = original_working_dir[:-len(f"\\{dataset_name}")]
        
        reconfigure_logger_for_method_dataset(
            method_name=method_name,
            dataset_name=dataset_name,
            exp_name=config_instance.exp_name,
            working_dir=original_working_dir
        )
        
        return config_instance
    
    @classmethod
    def default(cls):
        """Load default config
        - Priority: env < default_config_paths
        - Inside default_config_paths, the latter one overwrites the former one
        """
        default_config_paths: List[Path] = [
            GRAPHRAG_ROOT / "Option/Config2.yaml",
            CONFIG_ROOT / "Config2.yaml",
        ]

        dicts = [dict(os.environ)]
        dicts += [Config.read_yaml(path) for path in default_config_paths]

        final = merge_dict(dicts)
   
        return Config(**final)

    @property
    def extra(self):
        return self._extra

    @extra.setter
    def extra(self, value: dict):
        self._extra = value

    def get_openai_llm(self) -> Optional[LLMConfig]:
        """Get OpenAI LLMConfig by name. If no OpenAI, raise Exception"""
        if self.llm.api_type == LLMType.OPENAI:
            return self.llm
        return None
def parse(opt_path):
    
        with open(opt_path, mode='r') as f:
            opt = YamlModel.read_yaml(opt_path)
        # export CUDA_VISIBLE_DEVICES
        # gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
        # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
        return opt
def merge_dict(dicts: Iterable[Dict]) -> Dict:
    """Merge multiple dicts into one, with the latter dict overwriting the former"""
    result = {}
    for dictionary in dicts:
        result.update(dictionary)
    return result


default_config = Config.default() # which is used in other files, only for LLM, embedding and save file. 