# divide 부분 돌리고 난 다음 description_generate의 multi_process.py를 먼저 돌려줘야함
# 일단 새로운 가상 환경을 만듬
# conda create -n patch_matters_aggregation python=3.9 -y

# conda activate patch_matters_aggregation

# pip install torch torchvision transformers

# pip install icecream

# pip install shapely

# pip install "ms-swift[llm]==2.6.1"
# pip install "transformers==4.47.1"
# apt update && apt install -y git-lfs
# git lfs install

# bash run.sh

 sed -i '/"vision_feature_select_strategy": "default",/a\  "multimodal_projector_bias": false,' /root/.cache/modelscope/hub/models/swift/llava-v1.6-vicuna-7b-hf/config.json

 grep "multimodal_projector_bias" /root/.cache/modelscope/hub/models/swift/llava-v1.6-vicuna-7b-hf/config.json

sed -i 's/"multimodal_projector_bias": false/"multimodal_projector_bias": true/' /root/.cache/modelscope/hub/models/swift/llava-v1.6-vicuna-7b-hf/config.json

mkdir -p /root/coco_image
ln -s /root/patch_matters_re-6/coco_image/coco_sample_data_Image_Textualization /root/coco_image/coco_sample_data_Image_Textualization
mkdir -p /root/patch_matters_re-6/temp_image

#### 코드 추가 in multi_process.py
# 수정1
from transformers import LlamaTokenizerFast
LlamaTokenizerFast.max_token_id = property(
    lambda self: max(v for v in self.get_vocab().values() if v is not None)
)

# 수정2
img_src = '/root/patch_matters_re-6/coco_image/coco_sample_data_Image_Textualization/' + os.path.basename(key['image'])

# img_src = '../coco_image/coco_sample_data_Image_Textualization/'+key['image'].split('/')[-1]

# 수정 3
   start_time = time.time()
    model_type = 'llava1_6-vicuna-7b-instruct'
    # llm_engine = get_vllm_engine(model_type, torch.float16)
    model_dir = "/root/.cache/modelscope/hub/models/swift/llava-v1___6-vicuna-7b-hf"

    llm_engine = get_vllm_engine(model_type="llava1_6-vicuna-7b-instruct", model_dir=model_dir, dtype=torch.float16,  gpu_memory_utilization=0.7)
    # llm_engine = get_vllm_engine(model_type=model_type, model_dir=model_dir, dtype=torch.float16)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)
    llm_engine.generation_config.max_new_tokens = 512

##### 이후 실행
python /root/patch_matters_re-6/description_generate/multi_process.py \
  --input_file /root/patch_matters_re-6/description_generate/test_data/did_bench.json \
  --output_folder /root/patch_matters_re-6/description_generate/ \
  --chunk_index 0 \
  --chunk_num 1 \
  --node_index 0 \
  --node_num 1

### 그리고 결과 병합
python /root/patch_matters_re-6/description_generate/combine.py \
  --folder_path /root/patch_matters_re-6/description_generate \
  --output_file /root/patch_matters_re-6/description_generate/description_output.json


###기타 명령어#########################################################
#  find / -type d -name "llava-v1.6-vicuna-7b-hf" 2>/dev/null
#  df -h /

# 이건 일단...  보류
pip install transformers==4.36.2
pip install tokenizers==0.21.0
conda install python=3.10
git clone https://huggingface.co/llava-hf/llava-v1.6-vicuna-7b-hf
