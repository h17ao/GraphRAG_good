export CUDA_VISIBLE_DEVICES=3

python main.py -opt Option/Method/HippoRAG.yaml -dataset_name hotpotqa --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/LightRAG.yaml -dataset_name hotpotqa --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/Dalk.yaml -dataset_name hotpotqa --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/GR.yaml -dataset_name hotpotqa --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/KGP.yaml -dataset_name hotpotqa --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter

# python main.py -opt Option/Method/HippoRAG.yaml -dataset_name 2wiki --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/LightRAG.yaml -dataset_name 2wiki --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/Dalk.yaml -dataset_name 2wiki --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/KGP.yaml -dataset_name 2wiki --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/GR.yaml -dataset_name 2wiki --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter

python main.py -opt Option/Method/HippoRAG.yaml -dataset_name musique --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/LightRAG.yaml -dataset_name musique --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/Dalk.yaml -dataset_name musique --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/GR.yaml -dataset_name musique --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter
python main.py -opt Option/Method/KGP.yaml -dataset_name musique --retrieval_model qwen3-1.7b --retrieval_base_url http://localhost:8003/v1 --exp_name qwen3-1.7b-top-5-only-iter

