#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import defaultdict, Counter

def analyze_iterations_correctness(json_file):
    """分析JSON文件中不同iteration数量下的llm_eval_correct分布"""
    
    # 统计数据
    max_iterations = []  # 每个条目的最大iteration
    iteration_counts = Counter()  # iteration为1,2,3,4,5的条目数
    correctness_stats = defaultdict(lambda: {'correct_1.0': 0, 'correct_0.0': 0})  # 各iteration中correct 1.0/0.0的数量
    
    print(f"正在分析文件: {json_file}")
    print("="*60)
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                    line_count += 1
                    
                    # 获取llm_eval_correct值
                    llm_eval_correct = entry.get('llm_eval_correct', None)
                    if llm_eval_correct is None:
                        continue
                    
                    # 获取output_all数组
                    output_all = entry.get('output_all', [])
                    if not output_all:
                        continue
                    
                    # 找到该条目的最大iteration
                    iterations = []
                    
                    for item in output_all:
                        if 'iteration' in item:
                            iteration = item['iteration']
                            iterations.append(iteration)
                    
                    if iterations:
                        max_iter = max(iterations)
                        max_iterations.append(max_iter)
                        
                        # 统计该条目的最终iteration
                        if max_iter <= 5:
                            iteration_counts[max_iter] += 1
                            
                            # 统计该条目的llm_eval_correct值
                            if llm_eval_correct == 1.0:
                                correctness_stats[max_iter]['correct_1.0'] += 1
                            elif llm_eval_correct == 0.0:
                                correctness_stats[max_iter]['correct_0.0'] += 1
                        
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误在第{line_count}行: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"文件未找到: {json_file}")
        return
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return
    
    # 输出统计结果
    print(f"总共处理了 {line_count} 个条目")
    print(f"有iteration信息的条目: {len(max_iterations)} 个")
    print()
    
    # 统计最大iteration分布
    if max_iterations:
        max_iter_overall = max(max_iterations)
        print(f"整体最大iteration数: {max_iter_overall}")
        print()
        
        # 统计各iteration数量的条目数
        print("各iteration数量的条目统计:")
        print("-" * 40)
        for i in range(1, 6):
            count = iteration_counts.get(i, 0)
            percentage = (count / len(max_iterations) * 100) if max_iterations else 0
            print(f"iteration = {i}: {count:4d} 条目 ({percentage:5.1f}%)")
        
        # 统计超过5的iteration
        high_iter_count = sum(1 for x in max_iterations if x > 5)
        if high_iter_count > 0:
            high_iter_percentage = (high_iter_count / len(max_iterations) * 100)
            print(f"iteration > 5: {high_iter_count:4d} 条目 ({high_iter_percentage:5.1f}%)")
        
        print()
        
        # 统计各iteration中llm_eval_correct的分布
        print("各iteration中llm_eval_correct统计:")
        print("-" * 60)
        print("Iteration | Correct=1.0 | Correct=0.0 | Total | Accuracy")
        print("-" * 60)
        
        total_correct_1 = 0
        total_correct_0 = 0
        
        for i in range(1, 6):
            correct_1_count = correctness_stats[i]['correct_1.0']
            correct_0_count = correctness_stats[i]['correct_0.0']
            total = correct_1_count + correct_0_count
            
            total_correct_1 += correct_1_count
            total_correct_0 += correct_0_count
            
            if total > 0:
                accuracy = (correct_1_count / total * 100)
                print(f"    {i}     |    {correct_1_count:4d}     |    {correct_0_count:4d}     | {total:4d}  | {accuracy:6.1f}%")
            else:
                print(f"    {i}     |      0     |      0     |   0   |   0.0%")
        
        print("-" * 60)
        
        # 总体统计
        total_all = total_correct_1 + total_correct_0
        
        if total_all > 0:
            overall_accuracy = (total_correct_1 / total_all * 100)
            print(f"总体统计:")
            print(f"Total Correct=1.0: {total_correct_1} ({total_correct_1/total_all*100:.1f}%)")
            print(f"Total Correct=0.0: {total_correct_0} ({total_correct_0/total_all*100:.1f}%)")
            print(f"Overall Accuracy: {overall_accuracy:.1f}%")
            print(f"Total entries: {total_all}")
            
        print()
        
        # 按iteration分析准确率趋势
        print("准确率趋势分析:")
        print("-" * 30)
        for i in range(1, 6):
            correct_1_count = correctness_stats[i]['correct_1.0']
            correct_0_count = correctness_stats[i]['correct_0.0']
            total = correct_1_count + correct_0_count
            
            if total > 0:
                accuracy = (correct_1_count / total * 100)
                print(f"Iteration {i}: {accuracy:5.1f}% 准确率 (基于 {total} 个样本)")
            else:
                print(f"Iteration {i}: 无数据")

if __name__ == "__main__":
    # 分析KGP trained模型
    json_file = "/home/yh/GraphRAG_iter/hotpotqa/KGP_Hao/trained_qwen3-1.7b-top-5-only-iter/Results/llm_eval_results_09191122.json"
    print("=" * 80)
    print("KGP trained_qwen3-1.7b 模型分析")
    print("=" * 80)
    analyze_iterations_correctness(json_file)
    
    print("\n" + "=" * 80)
    print("KGP qwen3-1.7b 模型分析")
    print("=" * 80)
    # 分析KGP 原始模型
    json_file = "/home/yh/GraphRAG_iter/hotpotqa/KGP_Hao/qwen3-1.7b-top-5-only-/Results/llm_eval_results_09191116.json"
    analyze_iterations_correctness(json_file)
