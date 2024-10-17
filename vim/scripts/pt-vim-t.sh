#!/bin/bash

#python Vim/vim/main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-set 'CIFAR' --batch-size 16 --drop-path 0.0 --weight-decay 0.1 
python main.py --model vim_tiny_patch16_stride8_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --batch-size 128 --lr 5e-6 --min-lr 1e-5 --warmup-lr 1e-5 --drop-path 0.0 --weight-decay 1e-8 --num_workers 25 --data-path CIFAR  --epochs 30 

