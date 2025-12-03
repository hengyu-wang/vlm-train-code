1. 修改 ./LLaMA-Factory/data/coco_dataset_2014.json和./LLaMA-Factory/data/dataset_info.json 将~替换成当前目录
2. 添加 ./LLaMA-Factory/config/qwen2_5vl_sft.yaml 中添加qwen7b路径

cd ./LLaMA-Factory
pip install -e ".[torch,metrics]"

cd ../transformers
pip install -e .

cd ../LLaMA-Factory
最后运行：llamafactory-cli train config/qwen2_5vl_sft.yaml

