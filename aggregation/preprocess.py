# import json
# import os

# desc_path = "/root/patch_matters_re-13/description_generate/description_output.json"
# boxinfo_path = "/root/patch_matters_re-13/ovdet/main_box_progress.json"
# output_path = "/root/patch_matters_re-13/aggregation/description_with_box.json"


# with open(desc_path, 'r') as f:
#     desc_data = json.load(f)

# with open(boxinfo_path, 'r') as f:
#     box_data = json.load(f)

# box_lookup = {
#     os.path.basename(k): v["main_box"]
#     for k, v in box_data.items()
#     if "main_box" in v
# }

# for entry in desc_data:
#     img_name = os.path.basename(entry["image"])
#     entry["main_box"] = box_lookup.get(img_name, None)

# with open(output_path, 'w') as f:
#     json.dump(desc_data, f, indent=2)

# print(f"Merged output saved to {output_path}")


import json
import os

desc_path = "/root/patch_matters_re-13/description_generate/description_output.json"
boxinfo_path = "/root/patch_matters_re-13/ovdet/main_box_progress.json"
output_path = "/root/patch_matters_re-13/aggregation/description_output_with_box.json"

with open(desc_path, 'r') as f:
    desc_data = json.load(f)

with open(boxinfo_path, 'r') as f:
    box_data = json.load(f)

box_lookup = {
    os.path.basename(k): {
        "main_box": v.get("main_box", None),
        "four_box": v.get("equal_four_box", None)  # ← 여기서 key도 바꿔서 넣어줌
    }
    for k, v in box_data.items()
}

for entry in desc_data:
    img_name = os.path.basename(entry["image"])
    matched = box_lookup.get(img_name, {})
    entry["main_box"] = matched.get("main_box", None)
    entry["four_box"] = matched.get("four_box", None)

with open(output_path, 'w') as f:
    json.dump(desc_data, f, indent=2)

print(f"[✔] Merged output saved to {output_path}")
