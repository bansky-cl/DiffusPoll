export CUDA_VISIBLE_DEVICES=0,1 && python -m torch.distributed.launch --nproc_per_node=2 --master_port=12233 --use_env run_train.py \
--diff_steps 1000 \
--lr 0.0001 \
--learning_steps 80000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--dataset wbp \
--data_dir datasets/wbp {or your own path-to-datasets} \
--seq_len 256 \
--schedule_sampler lossaware \
--notes wbp-train \
--mask 0 \
--div_loss 1 \
--hidden_dim 128 \
--bsz 512 \
--dataset wbp \
--data_dir datasets/wbp{can choose different datasets} \
--vocab t5 \
--add_token 1 \
--microbatch 128 \
--config_name bert-base-chinese-m{needs modify the huggingface config in bert-base-chinese}
