export CUDA_VISIBLE_DEVICES=2


python main.py -opt Option/Method/HippoRAG.yaml -dataset_name 2wiki
python main.py -opt Option/Method/LightRAG.yaml -dataset_name 2wiki
python main.py -opt Option/Method/Dalk.yaml -dataset_name 2wiki
python main.py -opt Option/Method/GR.yaml -dataset_name 2wiki
python main.py -opt Option/Method/KGP.yaml -dataset_name 2wiki

# python main.py -opt Option/Method/HippoRAG_Hao.yaml -dataset_name 2wiki
# python main.py -opt Option/Method/LightRAG_Hao.yaml -dataset_name 2wiki
# python main.py -opt Option/Method/Dalk_Hao.yaml -dataset_name 2wiki
# python main.py -opt Option/Method/KGP_Hao.yaml -dataset_name 2wiki
# python main.py -opt Option/Method/GR_Hao.yaml -dataset_name 2wiki

# python main.py -opt Option/Method/HippoRAG.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/LightRAG.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/Dalk.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/GR.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/KGP.yaml -dataset_name hotpotqa


