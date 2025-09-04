#!/usr/bin/env python3
"""
从 hippo_rl_data.json 的 retrieved_context 中提取图数据
最终修复版本：使用与GraphRAG相同的chunk_id计算方式
"""

import json
from hashlib import md5
import logging
import networkx as nx
from typing import List, Dict, Any
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalGraphDataExtractor:
    def __init__(self, graph_data_path: str):
        """初始化图数据提取器"""
        self.graph_data_path = Path(graph_data_path)
        self.graph = None        # NetworkX graph
        
        logger.info(f"✅ 图数据提取器初始化完成，数据路径: {graph_data_path}")

    def load_graph(self):
        """加载GraphML图数据"""
        try:
            # 加载GraphML文件
            graphml_path = self.graph_data_path / "graph_storage_nx_data.graphml"
            if graphml_path.exists():
                self.graph = nx.read_graphml(str(graphml_path))
                logger.info(f"✅ GraphML数据加载成功: {self.graph.number_of_nodes()} 个节点, {self.graph.number_of_edges()} 条边")
                return True
            else:
                logger.warning(f"❌ GraphML文件不存在: {graphml_path}")
                return False
            
        except Exception as e:
            logger.error(f"❌ 图数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def mdhash_id(self, content: str, prefix: str = "") -> str:
        """
        使用与GraphRAG相同的哈希计算方式
        """
        return prefix + md5(content.encode()).hexdigest()

    def calculate_chunk_id(self, content: str) -> str:
        """
        根据内容计算chunk_id，使用与GraphRAG相同的方式
        """
        return self.mdhash_id(content, prefix="chunk-")

    def find_relations_by_chunks(self, chunk_ids: List[str]) -> List[Dict]:
        """
        根据chunk_ids查找相关关系（从GraphML图中）
        """
        if self.graph is None:
            logger.warning("Graph数据未加载")
            return []

        try:
            logger.info(f"🔍 查找 {len(chunk_ids)} 个chunks的相关关系")
            
            # 直接从图中查找包含chunk_id的边
            relations = []
            edges_data = self.graph.edges(data=True)
            
            chunk_id_set = set(chunk_ids)  # 转换为集合以提高查找效率
            
            for src, dst, edge_data in edges_data:
                source_id = edge_data.get('source_id', '')
                
                # 检查这条边是否来自我们的任何chunk
                # source_id可能包含多个chunk_id，用<SEP>分隔
                if source_id:
                    source_chunks = source_id.split('<SEP>')
                    
                    # 检查是否有任何chunk_id匹配
                    if any(chunk_id in chunk_id_set for chunk_id in source_chunks):
                        # 构造关系数据
                        relation = {
                            'src_id': edge_data.get('src_id', src),
                            'tgt_id': edge_data.get('tgt_id', dst),
                            'relation_name': edge_data.get('relation_name', 'related_to'),
                            'source_id': source_id,
                            'weight': edge_data.get('weight', 1.0)
                        }
                        relations.append(relation)
            
            logger.info(f"📊 成功从图中获取 {len(relations)} 个关系")
            return relations

        except Exception as e:
            logger.error(f"查找关系失败: {e}")
            import traceback
            traceback.print_exc()
            return []

    def find_entities_by_relations(self, relations: List[Dict]) -> List[Dict]:
        """
        根据关系查找相关实体（从GraphML图中）
        """
        if not relations or self.graph is None:
            return []

        entity_names = set()
        
        # 从关系中提取实体名
        for relation in relations:
            src_id = relation.get('src_id', '')
            tgt_id = relation.get('tgt_id', '')
            if src_id:
                entity_names.add(src_id)
            if tgt_id:
                entity_names.add(tgt_id)

        logger.info(f"📊 从 {len(relations)} 个关系中提取到 {len(entity_names)} 个唯一实体名")

        # 查找实体详细信息
        entities = []
        
        for entity_name in entity_names:
            if entity_name in self.graph:
                node_data = self.graph.nodes[entity_name]
                entity = {
                    'entity_name': entity_name,
                    'entity_type': node_data.get('entity_type', ''),
                    'source_id': node_data.get('source_id', ''),
                    'description': node_data.get('description', ''),
                    'content': node_data.get('content', entity_name)
                }
                entities.append(entity)
            else:
                # 如果图中没有找到，创建基本实体
                entities.append({
                    'entity_name': entity_name,
                    'entity_type': 'unknown'
                })

        logger.info(f"📊 成功获取 {len(entities)} 个实体详情")
        return entities

    def extract_graph_from_contexts(self, contexts: List[str]) -> Dict[str, Any]:
        """
        从retrieved_contexts提取图数据
        """
        logger.info(f"🔍 开始处理 {len(contexts)} 个上下文")

        # 1. 计算chunk_ids
        chunk_ids = []
        for i, context in enumerate(contexts):
            chunk_id = self.calculate_chunk_id(context)
            chunk_ids.append(chunk_id)
            logger.debug(f"上下文 {i+1}: {chunk_id}")

        logger.info(f"📋 计算得到 {len(chunk_ids)} 个chunk_ids")

        # 2. 查找相关关系
        relations = self.find_relations_by_chunks(chunk_ids)
        logger.info(f"📊 找到 {len(relations)} 个相关关系")

        # 3. 查找相关实体
        entities = self.find_entities_by_relations(relations)
        logger.info(f"📊 找到 {len(entities)} 个相关实体")

        # 4. 构建图结构
        return self._build_graph_structure(entities, relations)

    def _build_graph_structure(self, entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """
        构建图结构：将实体和关系转换为nodes和edges格式
        """
        # 构建实体ID映射
        entity_id_map = {}
        nodes_list = []
        
        for idx, entity in enumerate(entities):
            entity_id = idx + 1
            entity_name = entity.get('entity_name', f'Entity_{entity_id}')
            entity_id_map[entity_name] = entity_id
            
            nodes_list.append({
                "id": entity_id,
                "entity": entity_name
            })

        # 构建边
        edges_list = []
        for relation in relations:
            src_name = relation.get('src_id', '')
            dst_name = relation.get('tgt_id', '')
            relation_name = relation.get('relation_name', 'related_to')
            
            if src_name in entity_id_map and dst_name in entity_id_map:
                edges_list.append({
                    "src": entity_id_map[src_name],
                    "relation": relation_name,
                    "dst": entity_id_map[dst_name]
                })

        logger.info(f"✅ 图结构构建完成: {len(nodes_list)} 个节点, {len(edges_list)} 条边")
        
        return {
            "nodes_list": nodes_list,
            "edges_list": edges_list
        }

    def process_data_file(self, input_file: str, output_file: str, start_idx: int = 0, end_idx: int = None):
        """
        处理整个数据文件
        """
        logger.info(f"📂 开始处理文件: {input_file}")

        # 加载图数据
        if not self.load_graph():
            logger.error("❌ 图数据加载失败，无法继续")
            return

        # 读取数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        total_items = len(data)
        logger.info(f"📊 总共 {total_items} 条数据")

        # 确定处理范围
        if end_idx is None:
            end_idx = total_items
        end_idx = min(end_idx, total_items)
        start_idx = max(0, start_idx)

        logger.info(f"🎯 处理范围: {start_idx} - {end_idx}")

        # 处理数据
        for i in range(start_idx, end_idx):
            item = data[i]
            question = item.get('question', 'N/A')[:50]
            logger.info(f"\n🔄 处理条目 {i+1}/{total_items}: {question}...")

            # 提取retrieved_context
            retrieved_context = item.get('retrieved_context', [])

            if not retrieved_context:
                logger.warning(f"条目 {i+1} 没有retrieved_context，跳过")
                item['graph_data'] = {"nodes_list": [], "edges_list": []}
                continue

            # 提取图数据
            try:
                graph_data = self.extract_graph_from_contexts(retrieved_context)
                item['graph_data'] = graph_data

                nodes_count = len(graph_data['nodes_list'])
                edges_count = len(graph_data['edges_list'])
                logger.info(f"✅ 条目 {i+1} 完成: {nodes_count} 实体, {edges_count} 关系")

            except Exception as e:
                logger.error(f"❌ 条目 {i+1} 处理失败: {e}")
                import traceback
                traceback.print_exc()
                item['graph_data'] = {"nodes_list": [], "edges_list": []}

            # 每处理10条保存一次
            if (i + 1) % 10 == 0:
                logger.info(f"💾 中间保存: 已处理 {i + 1} 条")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

        # 最终保存
        logger.info(f"💾 最终保存到: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("🎉 处理完成！")

def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='从retrieved_context提取现有图数据')
    parser.add_argument('--input', default='/data/workspace/hippo_rl_data.json', 
                       help='输入JSON文件路径')
    parser.add_argument('--output', default='/data/workspace/hippo_rl_data_with_graph_final.json',
                       help='输出JSON文件路径')
    parser.add_argument('--graph-data', default='/data/workspace/hotpotqa_all/HippoRAG/er_graph_colbert',
                       help='图数据目录路径')
    parser.add_argument('--start', type=int, default=0, help='开始索引')
    parser.add_argument('--end', type=int, default=None, help='结束索引')
    parser.add_argument('--sample', type=int, default=None, help='仅处理前N条数据（用于测试）')

    args = parser.parse_args()

    # 如果指定了sample，设置end
    if args.sample:
        args.end = min(args.sample, args.end) if args.end else args.sample

    # 初始化提取器
    extractor = FinalGraphDataExtractor(args.graph_data)

    # 处理文件
    extractor.process_data_file(
        input_file=args.input,
        output_file=args.output,
        start_idx=args.start,
        end_idx=args.end
    )

if __name__ == "__main__":
    main()