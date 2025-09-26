export CUDA_VISIBLE_DEVICES=3

# python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name hotpotqa --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/LightRAG_Hao.yaml -dataset_name hotpotqa --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/Dalk_Hao.yaml -dataset_name hotpotqa --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/KGP_Hao.yaml -dataset_name hotpotqa --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name 2wiki --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/LightRAG_Hao.yaml -dataset_name 2wiki --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/Dalk_Hao.yaml -dataset_name 2wiki --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
python main.py -opt Option/Method/KGP_Hao.yaml -dataset_name 2wiki --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8002/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter

python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name musique --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/LightRAG_Hao.yaml -dataset_name musique --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/Dalk_Hao.yaml -dataset_name musique --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
# python main.py -opt Option/Method/KGP_Hao.yaml -dataset_name musique --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter

python main.py -opt Option/Method/GR_Hao.yaml -dataset_name hotpotqa --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
python main.py -opt Option/Method/GR_Hao.yaml -dataset_name 2wiki --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
python main.py -opt Option/Method/GR_Hao.yaml -dataset_name musique --retrieval_model trained1.7b_70b_step200 --retrieval_base_url http://localhost:8003/v1 --exp_name trained1.7b_70b_step200-top-5-only-iter
