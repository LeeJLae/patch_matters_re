# divide 부분 돌리고 난 다음 description_generate의 multi_process.py를 먼저 돌려줘야함
# 일단 새로운 가상 환경을 만듬
# conda create -n patch_matters_aggregation python=3.9 -y

# conda activate patch_matters_aggregation

# pip install torch torchvision transformers

# pip install icecream

# pip install shapely

# pip install ms-swift==2.5.2.post1

# pip install "vllm>=0.2.0"
(활용해야하는데.. 빈 문서임..!! 
https://github.com/modelscope/ms-swift/blob/main/swift/llm/inference_vllm.py)
# bash /root/patch_matters_re-1/description_generate/run.sh