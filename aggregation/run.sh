set -ex 

# export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
node_index=0
node_num=1
chunk_num=1


for (( chunk_index=0; chunk_index<=$[$chunk_num-1]; chunk_index++ ))
do
    # Assign each process to a unique GPU using chunk_index
    gpu_id=$chunk_index  # Assuming chunk_index corresponds to a GPU (e.g., 0, 1, 2 for 3 GPUs)
    
    # Set CUDA_VISIBLE_DEVICES to assign a unique GPU to each process
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python3 /root/patch_matters_re-13/aggregation/main.py \
        --input_data /root/patch_matters_re-13/aggregation/description_output_with_box.json \
        --output_folder /root/patch_matters_re-13/aggregation \
        --chunk_index $chunk_index \
        --chunk_num $chunk_num \
        --node_index $node_index \
        --node_num $node_num > /root/patch_matters_re-13/aggregation/$chunk_index.log 2>&1 &
done

wait

python3 /root/patch_matters_re-13/aggregation/combine.py \
    --folder_path /root/patch_matters_re-13/aggregation \
    --output_file /root/patch_matters_re-13/aggregation/aggregation_output.json