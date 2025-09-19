export CUDA_VISIBLE_DEVICES=1

python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name hotpotqa --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/LightRAG_Hao.yaml -dataset_name hotpotqa --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/Dalk_Hao.yaml -dataset_name hotpotqa --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/GR_Hao.yaml -dataset_name hotpotqa --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/KGP_Hao.yaml -dataset_name hotpotqa --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter

python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name 2wiki --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/LightRAG_Hao.yaml -dataset_name 2wiki --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/Dalk_Hao.yaml -dataset_name 2wiki --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/KGP_Hao.yaml -dataset_name 2wiki --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/GR_Hao.yaml -dataset_name 2wiki --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter

python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name musique --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/LightRAG_Hao.yaml -dataset_name musique --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/Dalk_Hao.yaml -dataset_name musique --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/GR_Hao.yaml -dataset_name musique --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
python main.py -opt Option/Method/KGP_Hao.yaml -dataset_name musique --retrieval_model qwen3-4b --retrieval_base_url http://localhost:8001/v1 --exp_name qwen3-4b-top-5-only-iter
