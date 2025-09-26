import pandas as pd
from torch.utils.data import Dataset
import os
import json

class RAGQueryDataset(Dataset):
    def __init__(self,data_dir):
        super().__init__()
      
        self.corpus_path = os.path.join(data_dir, "Corpus.json")
        self.qa_path = os.path.join(data_dir, "Question.json")
        self.dataset = self._read_json_file(self.qa_path)

    #   支持读取json和jsonl文件
    def _read_json_file(self, file_path):
        """智能读取JSON文件，支持标准JSON数组格式和JSONL格式"""
        try:
            # 首先尝试标准的JSON数组格式
            return pd.read_json(file_path, orient="records")
        except ValueError as e:
            if "Trailing data" in str(e) or "Extra data" in str(e):
                # 如果是"Trailing data"错误，可能是JSONL格式，尝试逐行读取
                print(f"标准JSON格式读取失败，尝试JSONL格式: {file_path}")
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # 跳过空行
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                return pd.DataFrame(data)
            else:
                raise e

    def get_corpus(self):
        corpus = self._read_json_file(self.corpus_path)
        corpus_list = []
        
        #   智能检测内容字段名（可能是"text"或"context"）
        content_field = "text" if "text" in corpus.columns else "context"
        
        for i in range(len(corpus)):
            corpus_list.append(
                {
                    "title": corpus.iloc[i]["title"],
                    "content": corpus.iloc[i][content_field],
                    "doc_id": i,
                }
            )
        return corpus_list

    def __len__(self):
        return len(self.dataset)

    #   添加并发查询处理(50并发)
    def __getitem__(self, idx):
        # 处理切片操作
        if isinstance(idx, slice):
            # 获取切片范围内的所有数据
            start, stop, step = idx.indices(len(self.dataset))
            result = []
            for i in range(start, stop, step):
                question = self.dataset.iloc[i]["question"]
                answer = self.dataset.iloc[i]["answer"]
                other_attrs = self.dataset.iloc[i].drop(["answer", "question"])
                result.append({"id": i, "question": question, "answer": answer, **other_attrs})
            return result
        else:
            # 处理单个索引
            question = self.dataset.iloc[idx]["question"]
            answer = self.dataset.iloc[idx]["answer"]
            other_attrs = self.dataset.iloc[idx].drop(["answer", "question"])
            return {"id": idx, "question": question, "answer": answer, **other_attrs}


if __name__ == "__main__":
    corpus_path = "tmp.json"
    qa_path = "tmp.json"
    query_dataset = RAGQueryDataset(qa_path=qa_path, corpus_path=corpus_path)
    corpus = query_dataset.get_corpus()
    print(corpus[0])
    print(query_dataset[0])
