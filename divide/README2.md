## 0단계 : 기본 Setting (Vessl) 
(1) pip3 install packaging
(2) pdf 25p : Create SSH key 부분
(3) pdf 25p : Add SSH key to your VESSL account 부분
(4) vessl workspace vscode

## 📦 1단계: Miniconda 설치

### 1. Miniconda 설치 스크립트 다운로드
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

### 2. 설치 스크립트 실행
```bash
bash Miniconda3-latest-Linux-x86_64.sh
```
#### 설치 중 선택 사항:
* 라이선스 동의: `yes`
* 설치 경로: 기본값 (`/root/miniconda3` 등)
* `conda init` 실행 여부: `yes` 권장

## 🌀 2단계: 쉘 초기화
설치 직후에는 다음 명령어로 환경 적용:
```bash
source ~/.bashrc
```
만약 여전히 `conda` 명령이 안 된다면:
```bash
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
```

## Installation NEW
conda create -n patch_matters_39 python=3.9 -y
conda activate patch_matters_39

# PyTorch 1.13.1 + cu116 설치 (Python 3.9용 휠 사용)
pip install https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp39-cp39-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp39-cp39-linux_x86_64.whl
pip install torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

# 그 다음에 requirements.txt 설치
pip install -r requirements.txt

# ########################################################################################

## Usage_ parameter 파일이랑 json 파일은 로컬에 다운 받은 다음, explorer에 드래그
# 불편하지만 아직까진 workplace reset되면 반복해야함

`ovdet/checkpoints`
- clip_vitb32.pth
- res50_fpn_soco_star_400.pth
- this_repo_R-50-FPN_CLIP_iter_90000.pth

`ovdet/data/coco/annotations`
- instances_train2017.json
- instances_val2017.json

`ovdet/data/coco/wusize`
- captions_train2017_tags_allcaps.json
- instances_train2017_base.json
- instances_train2017_novel.json
- instances_val2017_base.json
- instances_val2017_novel.json


### 이제... Divide Inference는 2가지임... generate_four_box.py & get_main_box.py
# ####################Patch Matters Divide 환경 구축 및 generate\_four\_box 실행 가이드
# 아까 위에서 만든 가상환경과 별도로 이 2개 파일을 실행하는거 에서도 별도 가상환경을 만들어야함함

## 📦 1. Conda 환경 구성
```bash
conda create -n patch_matters_divide python=3.8.19 -y
conda activate patch_matters_divide
```

## ⚙️ 2. PyTorch 및 CUDA 호환 버전 설치

```bash
# 본인의 GPU 드라이버와 호환되는 CUDA 버전 확인 필요
# 예시: CUDA 11.6을 사용할 경우 (GPU 없는 경우에도 설치는 필요함)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --index-url https://download.pytorch.org/whl/cu116
```

> ❗ **주의**: 로컬에서 GPU가 없을 경우 CUDA 디바이스 오류 발생함. 학습/추론은 서버에서 진행 권장.

---

## 📂 3. 필수 라이브러리 설치

```bash
# mmcv는 아래와 같이 CUDA, torch 버전에 맞게 설치
pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html
pip install mmdet==3.1.0
pip install mmengine==0.10.1

# 이건 한번만 하면 됨
apt update
apt install -y libgl1 

# 설치 확인
python -c "from mmdet.apis import DetInferencer; print('✅ DetInferencer import OK')"

# 기타 필수 라이브러리
pip install opencv-python-headless rich pillow tqdm
pip install ftfy


## 📁 4. 레포 구성

```
patchmatters-vessl/
├── divide/
│   ├── configs/
│   ├── checkpoints/  ← ✅ 학습된 모델 weight 저장
│   ├── data/         ← ✅ 결과 json 저장 폴더 (없으면 생성 필요)
│   ├── ovdet/        ← ✅ custom 모델 정의
│   ├── sample_tools/   # 걍 tools
│   ├── true_box_sample.py
│   └── generate_four_box.py
├── coco_image/
│   └── coco_sample_data_Image_Textualization/ ← ✅ 추론할 이미지 위치
```

## ✏️ 5. generate\_four\_box.py 코드 수정 사항

### 1. argparse → `arg` 대신 `args`로 전면 수정:

```python
arg = argparse.ArgumentParser()
...
args = arg.parse_args()
```

**수정 항목들:**

* `arg.image_folder → args.image_folder`
* `arg.four_box_save_path → args.four_box_save_path`
* `arg.object_box_save_path → args.object_box_save_path`
* 기타 `arg.`로 되어있던 모든 변수 → `args.`로 교체

### 2. GPU가 없는 경우 에러 방지용 디바이스 명시

```python
inference = DetInferencer(model=args.model_config_file, weights=args.checkpoint_file, device='cpu')
```
> ⚠️ `argparse`에서 `--device`를 인자로 받지 않기 때문에, 디바이스는 코드 내에서 직접 `device='cpu'`로 명시해야 함

---
# 그외 inference 하면서 몇 개의 추가 수정을 진행함
# 지금 생각나는건.... divide/ovdet/models/roi_heads/baron_bbox_heads/bbox_head.py
# 
## ▶️ 6. 실행 명령어 (로컬 or 서버)

```bash
python /root/patch_matters_re-1/divide/generate_four_box.py \
  --model_config_file "/root/patch_matters_re-1/divide/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py" \
  --checkpoint_file "/root/patch_matters_re-1/divide/ovdet/checkpoints/iter_90000.pth" \
  --image_folder "/root/patch_matters_re-1/coco_image/coco_sample_data_Image_Textualization" \
  --four_box_save_path "/root/patch_matters_re-1/divide/checkpoints/four_box.json" \
  --object_box_save_path "/root/patch_matters_re-1/divide/checkpoints/object_box.json"

```
# ##############################################################################################################################
# 불편편하지만 다시 실행할 떄 아래와 같이 진행을 반복했음 
### 1. Miniconda 설치 스크립트 다운로드
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh
source /root/miniconda3/etc/profile.d/conda.sh
source ~/.bashrc

conda activate patch_matters_divide
```

# ##############################################################################################################################

# #######################################################  get_main_box.py 실행 가이드
📂 1. 사전 준비
generate_four_box.py 실행 완료 후 아래 두 파일이 이미 존재해야 함
object_box.json
 -> object_box_save_path "/root/patch_matters_re-1/divide/checkpoints/object_box.json
이미지 폴더 (your image folder)
 -> image_folder "/root/patch_matters_re-1/coco_image/coco_sample_data_Image_Textualization"

✏️ 2. get_main_box.py 코드 확인 사항
코드 내에 arg. 으로 되어있는거 args.로 바꿔줘야함

# 그리고 아래와 같이 변경
arg.add_argument('--llm_path', type=str, help='LLM model', default='cache/huggingface/hub/mate-llama-3.1-8b-instruct')
-> arg.add_argument('--llm_path', type=str, help='LLM model', default='meta-llama/Meta-Llama-3-8B-Instruct')

⚠️ 3. 추가로 넣은 코드 (가상환경 내 설치 필요)
pip install transformers
pip install icecream
pip install 'accelerate>=0.26.0'
huggingface-cli login
# 중간에 토큰입력하라고 하는데, 그 값은 아래 방식으로 가져옴옴
이후 아래 링크로 이동하여 토큰 생성:
👉 https://huggingface.co/settings/tokens
"New token" 생성
이름: 예) llama-access
권한: ✅ 최소 read 권한 이상
생성된 토큰을 복사해서 지금 터미널에 붙여넣고 Enter

pip install --upgrade jinja2
pip install --upgrade transformers
mkdir -p ovdet


############################# get_main_box.py에서 추가 코드 수정1
# def generate_description(image_path, model, vis_processors, prompt=None):
#     image = Image.open(image_path).convert("RGB")
#     inputs = vis_processors(images=image, return_tensors="pt").to(model.device, torch.float16)
#     generated_ids = model.generate(**inputs
#                                    )
#     generated_text = vis_processors.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
#     return generated_text
###############################################################################################
def generate_description(image_path, model, processor):
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")

    prompt = "Describe this image in one sentence."

    # 이미지와 텍스트 프롬프트 동시 입력
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, dtype=torch.float16)

    # generate 호출
    generated_ids = model.generate(**inputs, max_new_tokens=30)

    # 텍스트 디코딩
    generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

############################# get_main_box.py에서 추가 코드 수정2
# def re_match(input_string):
#     matched_content = re.findall(r"\[.*?\]", input_string)
#     cleaned_items = matched_content[0][1:-1].replace("'", "").strip()


#     return [cleaned_items] if "," not in cleaned_items else [item.strip() for item in cleaned_items.split(",")]
#################################################################################################################
def re_match(common_objects):
    pattern = r"\[(.*?)\]"
    matched_content = re.findall(pattern, str(common_objects))
    if not matched_content:
        return []  # 또는 return common_objects, 필요에 따라 조정
    cleaned_items = matched_content[0].replace("'", "").strip()
    return [item.strip() for item in cleaned_items.split(",") if item.strip()]
#################################################################################################################

▶️ 4. 실행 코드
python /root/patch_matters_re-1/divide/get_main_box.py \
  --image_folder /root/patch_matters_re-1/coco_image/coco_sample_data_Image_Textualization \
  --object_box_save_path /root/patch_matters_re-1/divide/checkpoints/object_box.json \
  --main_box_save_path divide/data/main_box.json



# 여기까지 했으면 divide 부분은 완료!! 이제 aggregation 부분을 해야함함