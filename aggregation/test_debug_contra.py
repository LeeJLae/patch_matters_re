# # #################################
# # contra 생성 책임자는 semantic_batch.py의 batch_merge_main()
# # 그런데 batch_merge_main()이 제대로 된 값을 못 만들면 contra가 None이 됨 → 결국 결과에 안 찍힘
# # 따라서 batch_merge_main() 내부에서 실제 어떤 조건에서 contra를 만들지 못하고 있는지가 진짜 핵심 디버깅 포인트
# # # ###################################
# # categories['For Contradictory Triples'] 추출 성공 여부 먼저 print
# # → print("contra group:", categories['For Contradictory Triples'])

# # merge 값이 실제로 유효한 이미지인지 확인
# # → print(merge.size) 또는 type(merge) 넣어서 체크

# # score가 실제로 나오는지, 0.3 이상 있는지 출력
# # → print("similarity_contra_list:", similarity_contra_list)
# # # ###################################

# # 에러 원인은 여기입니다:

# # sql
# # 복사
# # 편집
# # max_contra = max(contra)  
# # TypeError: '>' not supported between instances of 'NoneType' and 'NoneType'
# # 즉, contra 리스트의 값이 전부 None이라 max() 연산이 불가능해진 상황입니다.
# # # ###################################


# import os
# import json
# from PIL import Image

# # 설정
# input_json_path = '/root/patch_matters_re-18/aggregation/description_output_with_box.json'  # 네 input json 경로
# image_folder = '/root/patch_matters_re-18/coco_image/coco_sample_data_Image_Textualization'  # 이미지들이 저장된 폴더
# output_path = '/root/patch_matters_re-18/aggregation/result/contra_debug_output.json'



# # JSON 로드
# with open(input_json_path, 'r') as f:
#     data = json.load(f)

# result = []

# for idx, item in enumerate(data):
#     if idx >= 5:  # 최대 5개까지만 확인
#         break

#     file_name = os.path.basename(item['image'])  # => "000000248382.jpg"
#     image_path = os.path.join(image_folder, file_name)

#     try:
#         image = Image.open(image_path).convert("RGB")
#     except Exception as e:
#         print(f"[ERROR] Cannot open image: {image_path} -> {e}")
#         continue

#     # dummy contra 값 예시: 이미지 사이즈 기준으로 간단하게 만들어봄
#     width, height = image.size
#     contra_list = [f"Size: {width}x{height}", f"File: {file_name}"]

#     result.append({
#         "image": item['image'],
#         "contra": contra_list
#     })
#     print(f"[INFO] Processed: {file_name} -> {contra_list}")

# # 저장
# with open(output_path, 'w') as f:
#     json.dump(result, f, indent=2)

# print(f"[DONE] Saved result to {output_path}")


import os
import json
from PIL import Image
from semantic_batch import fusion  # 네 코드 구조에 맞게 import 수정

# ───── Dummy 클래스들 ───── #
class DummyLLM:
    def generate(self, prompts, sampling_params):
        class Output:
            class Inner:
                text = "Dummy merged caption"
            outputs = [Inner()]
        return [Output()]

class DummyTokenizer:
    def eos_token_id(self): return 0
    def convert_tokens_to_ids(self, token): return 0
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "dummy prompt"

class DummyBlipModel:
    def rank_captions(self, image, candidates):
        return candidates  # 그냥 그대로 반환

# ───── 경로 설정 ───── #
input_json_path = '/root/patch_matters_re-18/aggregation/description_output_with_box.json'
image_folder = '/root/patch_matters_re-18/coco_image/coco_sample_data_Image_Textualization'
output_json_path = '/root/patch_matters_re-18/aggregation/result/batch_merge_contra_debug_output.json'

# ───── JSON 로드 ───── #
with open(input_json_path, 'r') as f:
    input_data = json.load(f)

input_data = input_data[:5]

# ───── Fusion 객체 생성 ───── #
dummy_llm = DummyLLM()
dummy_tokenizer = DummyTokenizer()
dummy_blip = DummyBlipModel()
fusion_instance = fusion(dummy_llm, dummy_tokenizer, dummy_blip, input_data)

# ───── 배치 입력 구성 ───── #
batch_input = []
results_to_save = []

for item in input_data:
    file_name = os.path.basename(item['image'])
    image_path = os.path.join(image_folder, file_name)

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Cannot open image: {image_path} -> {e}")
        continue

    dummy_region_dict = {'5': image}
    batch_input.append([(None, dummy_region_dict)])

# ───── batch_merge_main 실행 ───── #
output = fusion_instance.batch_merge_main(batch_input)

# ───── 결과 저장 ───── #
for idx, item in enumerate(input_data):
    results_to_save.append({
        "image": item['image'],
        "contra": output[idx] if output and idx < len(output) else None
    })

with open(output_json_path, 'w') as f:
    json.dump(results_to_save, f, indent=2)

print(f"[DONE] Saved batch_merge_main contra result to:\n{output_json_path}")
