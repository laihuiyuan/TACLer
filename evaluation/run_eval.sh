
for dataset in math aime2024 amc aime2025
do
  if [ "$dataset" = "math" ]; then
    sample_n=1
  else
    sample_n=16
  fi
  CUDA_VISIBLE_DEVICES='0' \
  python -m tacl.eval \
  --model_path $1 \
  --data_path "data/${dataset}.json" \
  --output_dir "outputs" \
  --temperature 0.6 \
  --top_p 0.95 \
  --sample_n $sample_n \
  --pass_k 1 \
  --max_tokens 16384
done


for dataset in math aime2024 amc aime2025
do
  if [ "$dataset" = "math" ]; then
    sample_n=1
  else
    sample_n=16
  fi
  CUDA_VISIBLE_DEVICES='0' \
  python -m tacl.eval \
  --model_path $1 \
  --think_mode \
  --data_path "data/${dataset}.json" \
  --output_dir "outputs" \
  --temperature 0.6 \
  --top_p 0.95 \
  --sample_n $sample_n \
  --pass_k 1 \
  --max_tokens 16384
done

