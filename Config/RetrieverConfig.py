from Core.Utils.YamlModel import YamlModel


class RetrieverConfig(YamlModel):
    # Retrieval Config
    query_type: str = "ppr"
    enable_local: bool = False
    use_entity_similarity_for_ppr: bool = True
    top_k_entity_for_ppr: int = 8
    node_specificity: bool = True
    damping: float = 0.1
    top_k: int = 5
    k_nei: int = 3
    node_specificity: bool = True
    damping: float = 0.1
    max_token_for_local_context: int = 4800  # maximum token  * 0.4
    max_token_for_global_context: int = 4000 # maximum token  * 0.3
    local_max_token_for_text_unit: int = 4000  # 12000 * 0.33
    use_relations_vdb: bool = False
    use_subgraphs_vdb: bool = False
    global_max_consider_community: int = 512
    global_min_community_rating: float = 0.0
    level: int = 2  # haloyang 添加Community level for filtering
