#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from Core.GraphRAG import GraphRAG
from Option.Config2 import Config
import argparse
import os
import asyncio
from pathlib import Path
from shutil import copyfile
from Data.QueryDataset import RAGQueryDataset
import pandas as pd
from Core.Utils.Evaluation import Evaluator
from Core.Utils.llm_evaluation import wrapper_llm_evaluation

def check_dirs(opt):
    # For each query, save the results in a separate directory
    result_dir = os.path.join(opt.working_dir, opt.exp_name, "Results")
    # Save the current used config in a separate directory
    config_dir = os.path.join(opt.working_dir, opt.exp_name, "Configs")
    # Save the metrics of entire experiment in a separate directory
    metric_dir = os.path.join(opt.working_dir, opt.exp_name, "Metrics")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metric_dir, exist_ok=True)
    opt_name = args.opt[args.opt.rindex("/") + 1 :]
    basic_name = args.opt.split("/")[0] + "/Config2.yaml"
    copyfile(args.opt, os.path.join(config_dir, opt_name))
    copyfile(basic_name, os.path.join(config_dir, "Config2.yaml"))
    return result_dir

#   修改为同时保存context且实时读写保存
def load_existing_results(result_dir):
    """断点续传：加载已有结果"""
    save_path = os.path.join(result_dir, "results.json")
    if os.path.exists(save_path):
        try:
            all_res = pd.read_json(save_path, lines=True).to_dict('records')
            start_idx = len(all_res)
            print(f"从第 {start_idx+1} 条继续...")
            return start_idx, all_res
        except:
            print("从头开始...")
    return 0, []

#   修改为同时保存context且实时读写保存
def save_results(all_res, result_dir):
    """保存结果到文件"""
    save_path = os.path.join(result_dir, "results.json")
    df = pd.DataFrame(all_res)
    df.to_json(save_path, orient="records", lines=True)
    return save_path

#   修改为同时保存context且实时读写保存
#   添加并发查询处理(50并发)
async def wrapper_query(query_dataset, digimon, result_dir, max_concurrent=50):
    """分批并发处理query，确保每批全部完成再进行下一批"""
    dataset_len = len(query_dataset)
    # 断点续传
    start_idx, all_res = load_existing_results(result_dir)
    remaining_queries = query_dataset[start_idx:]
    
    if not remaining_queries:
        print("所有查询已完成！")
        return save_results(all_res, result_dir)
    
    print(f"分批并发处理 {len(remaining_queries)} 个查询，每批{max_concurrent}个")
    
    # 分批处理
    for batch_start in range(0, len(remaining_queries), max_concurrent):
        batch_end = min(batch_start + max_concurrent, len(remaining_queries))
        batch_queries = remaining_queries[batch_start:batch_end]
        
        print(f"处理第 {batch_start//max_concurrent + 1} 批: {batch_start + start_idx + 1} - {batch_end + start_idx}")
        
        # 每批内部并发处理
        semaphore = asyncio.Semaphore(max_concurrent)
        batch_results = []
        
        async def process_single_query(query_item, idx_in_batch):
            async with semaphore:
                original_idx = start_idx + batch_start + idx_in_batch
                result = await digimon.query(query_item["question"])
                query_item["output"] = result["response"]
                query_item["retrieved_context"] = result["context"]
                query_item["id"] = original_idx
                
                # 保存HaoQuery的详细信息（如果存在）
                if "total_iterations" in result:
                    query_item["total_iterations"] = result["total_iterations"]
                if "historical_queries" in result:
                    query_item["historical_queries"] = result["historical_queries"]
                if "output_all" in result:
                    query_item["output_all"] = result["output_all"]
                
                print(f"{original_idx + 1}/{dataset_len}: 完成")
                return query_item
        
        # 执行当前批次的并发查询
        tasks = [process_single_query(query, i) for i, query in enumerate(batch_queries)]
        batch_results = await asyncio.gather(*tasks)
        
        # 将当前批次结果加入总结果
        all_res.extend(batch_results)
        
        # 保存当前进度
        save_path = save_results(all_res, result_dir)
        print(f"第 {batch_start//max_concurrent + 1} 批完成，已保存 {len(all_res)} 个结果")
    
    print(f"全部处理完成！共处理 {len(remaining_queries)} 个查询，结果已保存到: {save_path}")
    return save_path


async def wrapper_evaluation(path, opt, result_dir):
    from datetime import datetime
    
    # 生成时间戳 (月日小时分钟)
    timestamp = datetime.now().strftime("%m%d%H%M")
    
    # 执行标准评估
    print(f"📈 执行标准评估")
    eval = Evaluator(path, opt.dataset_name)
    res_dict = await eval.evaluate()
    
    # 带时间戳的metrics文件
    metrics_file = os.path.join(result_dir, f"metrics_{timestamp}.json")
    with open(metrics_file, "w") as f:
        f.write(str(res_dict))
    
    # 获取标准评估生成的.score.json文件路径
    score_file = path.replace(".json", "_score.json")
    print(f"标准评估完成，结果保存到: {score_file}")
    print(f"评估指标保存到: {metrics_file}")
    
    # 执行LLM评估
    print(f"执行LLM评估")
    # 使用Config中的eval_llm配置进行LLM评估
    await wrapper_llm_evaluation(path, config=opt)
    
    # LLM评估的结果文件路径（带时间戳）
    llm_eval_file = os.path.join(result_dir, f"llm_eval_results_{timestamp}.json")
    llm_summary_file = os.path.join(result_dir, f"llm_eval_summary_{timestamp}.json")
    print(f"LLM评估完成，结果文件已生成")


if __name__ == "__main__":

    # with open("./book.txt") as f:
    #     doc = f.read()

    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument("-dataset_name", type=str, help="Name of the dataset.")
    parser.add_argument("-method_name", type=str, help="Name of the method.", default=None)
    parser.add_argument("--retrieval_model", type=str, help="Override retrieval LLM model name", default=None)
    parser.add_argument("--retrieval_base_url", type=str, help="Override retrieval LLM base_url", default=None)
    parser.add_argument("--exp_name", type=str, help="Override experiment name", default=None)
    args = parser.parse_args()

    # 如果没有提供method_name，尝试从配置文件路径中自动提取
    method_name = args.method_name
    if method_name is None and args.opt:
        # 从路径如 "Option/Method/HippoRAG.yaml" 中提取 "HippoRAG"
        opt_path = Path(args.opt)
        if opt_path.parent.name == "Method":  # 确保是Method目录下的配置文件
            method_name = opt_path.stem  # 去掉扩展名的文件名
            print(f"自动从配置文件路径提取方法名: {method_name}")

    # 如果有exp_name参数，先临时修改环境变量，让Config.parse使用正确的exp_name
    original_exp_name = None
    if getattr(args, "exp_name", None):
        # 临时设置环境变量，让Config.parse使用命令行指定的exp_name
        import os
        original_exp_name = os.environ.get("EXP_NAME")
        os.environ["EXP_NAME"] = args.exp_name
        print(f"临时设置实验名称为: {args.exp_name}")

    opt = Config.parse(Path(args.opt), dataset_name=args.dataset_name, method_name=method_name)
    
    # 恢复原始环境变量
    if original_exp_name is not None:
        if original_exp_name:
            os.environ["EXP_NAME"] = original_exp_name
        else:
            os.environ.pop("EXP_NAME", None)

    # 覆盖检索模型名称（可选）
    if getattr(args, "retrieval_model", None):
        if not hasattr(opt, "retrieval_llm") or opt.retrieval_llm is None:
            raise ValueError("配置中未找到 retrieval_llm，请在 Option/Config2.yaml 中设置后再使用 --retrieval_model 覆盖")
        # 覆盖模型名
        opt.retrieval_llm.model = args.retrieval_model
        print(f"覆盖检索模型为: {opt.retrieval_llm.model}")

    # 覆盖检索模型的 base_url（可选）
    if getattr(args, "retrieval_base_url", None):
        if not hasattr(opt, "retrieval_llm") or opt.retrieval_llm is None:
            raise ValueError("配置中未找到 retrieval_llm，请在 Option/Config2.yaml 中设置后再使用 --retrieval_base_url 覆盖")
        opt.retrieval_llm.base_url = args.retrieval_base_url
        print(f"覆盖检索 base_url 为: {opt.retrieval_llm.base_url}")

    # exp_name 已在 Config.parse() 阶段通过环境变量处理
    digimon = GraphRAG(config=opt)
    result_dir = check_dirs(opt)

    query_dataset = RAGQueryDataset(
        data_dir=os.path.join(opt.data_root, opt.dataset_name)
    )
    corpus = query_dataset.get_corpus()
    # corpus = corpus[:10]

    asyncio.run(digimon.insert(corpus))

    # 检查是否需要进行查询处理
    results_file = os.path.join(result_dir, "results.json")
    start_idx, existing_results = load_existing_results(result_dir)
    
    if start_idx >= len(query_dataset):
        print(f"所有 {len(query_dataset)} 条查询已完成")
        print(f"跳过检索阶段，直接进入评估")
        save_path = results_file
    else:
        print(f"开始检索阶段：从第 {start_idx + 1} 条继续（共 {len(query_dataset)} 条）")
        save_path = asyncio.run(wrapper_query(query_dataset, digimon, result_dir, max_concurrent=50))

    # 执行评估
    print(f"开始评估阶段")
    asyncio.run(wrapper_evaluation(save_path, opt, result_dir))
    print("标准评估和LLM评估均已完成")

    # for train_item in dataloader:

    # a = asyncio.run(digimon.query("Who is Fred Gehrke?"))

    # asyncio.run(digimon.query("Who is Scrooge?"))
