# Patch Matters: Training-free Fine-grained Image Caption Enhancement via Local Perception
## 0단계 : 기본 Setting (Vessl) 
# (1) pip3 install packaging
# (2) pdf 25p : Create SSH key 부분
# (3) pdf 25p : Add SSH key to your VESSL account 부분
# (4) vessl workspace vscode

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

---

## 🌀 2단계: 쉘 초기화

설치 직후에는 다음 명령어로 환경 적용:

```bash
source ~/.bashrc
```

만약 여전히 `conda` 명령이 안 된다면:

```bash
eval "$(/root/miniconda3/bin/conda shell.bash hook)"
```

## Installation ##########################################
# 아래꺼는 3.8 환경으로 3.9로 대체함#######################
# 
# 
# conda create -n patch_matters python==3.8.19

# 가상환경 활성화화
# conda activate patch_matters 
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
# pip install -r requirements.txt
```
# #####################################################################
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

## Usage

### Checkpoints

google drive에 올려놨습니다. 다운받으시고 아래 경로 확인 후에 경로대로 넣으시면 됩니다.

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


### 이제... Divide Inference는 2가지임... 아래 2개를 다 돌려야함 
### 순차적으로 우선 재윤이가 해준 generate_four_box.py를 해보자자

1) python divide/generate_four_box.py --image_folder 'your image folder' --four_box_save_path 'four_box.json' --object_box_save_path 'object_box.json'

2) python ovdet/get_main_box.py --image_folder 'your image folder' --object_box_save_path 'object_box.json' --main_box_save_path 'main_box.json'

```


# ##############################################################################################################################
# ####################Patch Matters Divide 환경 구축 및 generate\_four\_box 실행 가이드 (w/ 시행착오 정리)

이 문서는 Patch Matters 프로젝트의 `divide` 환경을 처음부터 구축하고 `generate_four_box.py` 스크립트를 성공적으로 실행하기까지의 **모든 과정과 시행착오**를 기록한 문서입니다.

## 📦 1. Conda 환경 구성

```bash
conda create -n patch_matters_divide python=3.8.19 -y
conda activate patch_matters_divide
```

---

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

# 한번만 하면 됨
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
│   ├── sample_tools/
│   ├── true_box_sample.py
│   └── generate_four_box.py
├── coco_image/
│   └── coco_sample_data_Image_Textualization/ ← ✅ 추론할 이미지 위치
```

> **필수 파일**:
>
> * `divide/ovdet/checkpoints/iter_90000.pth`
> * `divide/ovdet/data/metadata/coco_clip_hand_craft_attn12.npy`
 -> (변경 필요)  divide/data/metadata/coco_clip_hand_craft_attn12.npy
---

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

## ▶️ 6. 실행 명령어 (로컬 or 서버)

```bash
python generate_four_box.py ^
  --model_config_file "C:/patchmatters-vessl/divide/configs/baron/ov_coco/baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py" ^
  --checkpoint_file "C:/patchmatters-vessl/divide/ovdet/checkpoints/iter_90000.pth" ^
  --image_folder "C:/patchmatters-vessl/coco_image/coco_sample_data_Image_Textualization" ^
  --four_box_save_path "C:/patchmatters-vessl/divide/data/four_box.json" ^
  --object_box_save_path "C:/patchmatters-vessl/divide/data/object_box.json"
```

---

## ⚠️ 7. 주요 시행착오 정리

### ✅ 체크포인트 파일 경로 오류

* `FileNotFoundError` 발생 시 경로 확인 필수
* 파일명 오타: `this_repo_R-50-FPN_CLIP_iter_90000.pth` vs `iter_90000.pth`

### ✅ config와 weight 간 불일치 오류

* config 파일은 반드시 `ovdet`과 맞춰야 함 (`baron_kd_faster_rcnn_r50_fpn_syncbn_90kx2.py`)

### ✅ custom registry 관련 warning

```
WARNING - Failed to search registry with scope "mmdet" in the "baron" registry tree.
```

* 이는 `ovdet` 내부에 정의된 custom registry 사용에 따른 것으로, 실제 inference 결과에 영향 없음

### ✅ inference 속도 매우 느림 (로컬)

* GPU 미탑재 시스템에서는 인퍼런스가 극도로 느림 → 서버에서 실행 권장

### ✅ CUDA 오류

```
RuntimeError: Found no NVIDIA driver on your system.
```

* 로컬에서 CUDA 디바이스를 사용하려 할 경우 발생. `device='cpu'`로 명시하거나 서버에서 실행 필요

### ✅ argparse 오타

```python
arg.image_folder → ❌
args.image_folder → ✅
```

### ✅ `.json` 저장이 안될 때

* `data/` 폴더 미생성 → 수동 생성 필요
* 터미널에서 파일 생성 여부 확인:

```bash
watch -n 2 ls -lh divide/data
```

---

## ✅ 성공 조건 체크

* 터미널 출력에 `image path is doing:`이 반복적으로 출력됨
* `four_box.json`, `object_box.json` 크기가 점점 커짐

# ##############################################################################################################################
# #######################################################  get_main_box.py 실행 가이드 (w/ 시행착오 정리)
📂 1. 사전 준비
generate_four_box.py 실행 완료 후 아래 두 파일이 이미 존재해야 함:
object_box.json
이미지 폴더 (your image folder)

✏️ 2. get_main_box.py 코드 확인 사항
args = arg.parse_args()  # ✅ 이미 고쳐져 있다고 가정
만약 args가 아니라 arg.object_box_save_path 식이면 → args.object_box_save_path로 수정해야 함

⚠️ 3. 주요 시행착오 예상 포인트
object_box_save_path의 경로가 실제로 존재하는지 확인
image_folder 경로가 정확한지 확인
main_box.json 저장될 폴더가 없으면 생성 필요
mkdir -p divide/data
▶️ 4. 실행 예시
python divide/ovdet/get_main_box.py \
  --image_folder divide/sample_image \
  --object_box_save_path divide/data/object_box.json \
  --main_box_save_path divide/data/main_box.json
✅ 성공 조건
콘솔에 이미지 경로 처리 로그가 뜸
main_box.json 파일 생성됨
에러 없이 완료


# 여기까지 했으면 divide 부분은 완료!! 이제 aggregation 부분을 해야함함