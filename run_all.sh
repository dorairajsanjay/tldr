export CUDA_VISIBLE_DEVICES=1
nohup python tldr_main.py --models_dir=models_new --logs_dir=logs_new > out_new.txt &
export CUDA_VISIBLE_DEVICES=2
nohup python tldr_main.py --inference_style=greedy_search --models_dir=models_baseline --logs_dir=logs_baseline > out_baseline.txt &
export CUDA_VISIBLE_DEVICES=3
nohup python tldr_main.py --mode=inference_only --inference_style=greedy_search --models_dir=models_baseline --logs_dir=logs2_baseline_inf > out_baseline_inf.txt&
