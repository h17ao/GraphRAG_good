# python main.py -opt Option/Method/Dalk.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/GR.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/HippoRAG.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/KGP.yaml -dataset_name hotpotqa
# python main.py -opt Option/Method/LightRAG.yaml -dataset_name hotpotqa


python main.py -opt Option/Method/HippoRAG.yaml -dataset_name 2wiki
python main.py -opt Option/Method/LightRAG.yaml -dataset_name 2wiki
python main.py -opt Option/Method/Dalk.yaml -dataset_name 2wiki
python main.py -opt Option/Method/GR.yaml -dataset_name 2wiki
python main.py -opt Option/Method/KGP.yaml -dataset_name 2wiki


python main.py -opt Option/Method/Dalk.yaml -dataset_name musique
python main.py -opt Option/Method/HippoRAG.yaml -dataset_name musique
python main.py -opt Option/Method/LightRAG.yaml -dataset_name musique
python main.py -opt Option/Method/GR.yaml -dataset_name musique
python main.py -opt Option/Method/KGP.yaml -dataset_name musique



python main.py -opt Option/Method/ToG.yaml -dataset_name hotpotqa
python main.py -opt Option/Method/ToG.yaml -dataset_name wiki
python main.py -opt Option/Method/ToG.yaml -dataset_name musique
python main.py -opt Option/Method/GGraphRAG.yaml -dataset_name hotpotqa
python main.py -opt Option/Method/GGraphRAG.yaml -dataset_name wiki
python main.py -opt Option/Method/GGraphRAG.yaml -dataset_name musique