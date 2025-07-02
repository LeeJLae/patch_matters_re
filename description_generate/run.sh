# set -ex 

# node_index=0
# node_num=1
# chunk_num=1

# # bash prepare.sh

# for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
# do
#     # Assign each process to a unique GPU using chunk_index
#     gpu_id=$chunk_index  # Assuming chunk_index corresponds to a GPU (e.g., 0, 1, 2 for 3 GPUs)
    
#     # Set CUDA_VISIBLE_DEVICES to assign a unique GPU to each process
#     CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 /root/patch_matters_re-9/description_generate/multi_process.py \
#         --input_file /root/patch_matters_re-9/description_generate/test_data/did_bench.json \
#         --output_folder /opt/output/description_generate \
#         --chunk_index $chunk_index \
#         --chunk_num $chunk_num \
#         --node_index $node_index \
#         --node_num $node_num > /root/patch_matters_re-9/description_generate/test_$chunk_index.log 2>&1 &
# done

# wait

# python3 /root/patch_matters_re-9/description_generate/combine.py \
#     --folder_path /opt/output/description_generate \
#     --output_file /root/patch_matters_re-9/description_generate/description_output.json 

##############inode 우회 ##################################

set -ex

export PYTHONPATH=/opt/pip_target:$PYTHONPATH

node_index=0
node_num=1
chunk_num=1

mkdir -p /opt/logs
mkdir -p /opt/output/description_generate

for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    gpu_id=$chunk_index
    
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 /root/patch_matters_re-10/description_generate/multi_process.py \
        --input_file /root/patch_matters_re-10/description_generate/test_data/did_bench.json \
        --output_folder /root/patch_matters_re-10/description_generate \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > /root/patch_matters_re-10/description_generate/test_$chunk_index.log 2>&1 &
done

wait

python3 /root/patch_matters_re-10/description_generate/combine.py \
    --folder_path /root/patch_matters_re-10/description_generate \
    --output_file /root/patch_matters_re-10/description_generate/description_output.json



