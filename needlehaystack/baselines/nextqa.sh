benchmark_name="nextqa"
cache_dir="./cache_dir"
GPT_Zero_Shot_QA="../VideoQA"
video_dir="${GPT_Zero_Shot_QA}/NExT_Zero_Shot_QA/videos"
gt_file_question="${GPT_Zero_Shot_QA}/NExT_Zero_Shot_QA/test_q.json"
gt_file_answers="${GPT_Zero_Shot_QA}/NExT_Zero_Shot_QA/test_a.json"
output_dir="results/NeXTQA"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

model_names=("PLLaVA-13B" "PLLaVA-34B" "LLaVA-NeXT-Video-34B")
environments=("pllava" "pllava" "llavanext")

for i in "${!model_names[@]}"; do
    model_name="${model_names[$i]}"
    environment="${environments[$i]}"
    source ~/scratch/anaconda3/bin/activate
    conda activate "$environment"
    export DECORD_EOF_RETRY_MAX=20480
    for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python ../evaluations/evaluation.py \
        --model_name ${model_name} \
        --benchmark_name ${benchmark_name} \
        --cache_dir ${cache_dir} \
        --video_dir ${video_dir} \
        --gt_file_question ${gt_file_question} \
        --gt_file_answers ${gt_file_answers} \
        --output_dir ${output_dir} \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX &
    done
    wait
    output_file=${output_dir}/merge.jsonl
    > "$output_file"
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ${output_dir}/${benchmark_name}_${model_name}_${IDX}.json >> "$output_file"
    done
    python ../evaluations/evaluate.py \
        --src $output_file \
        --model_name ${model_name} \
        --benchmark_name ${benchmark_name} \
        --output_dir ${output_dir} &
done