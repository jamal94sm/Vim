#!/bin/bash

python Vim/vim/main.py --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 --data-set 'CIFAR' --batch-size 16 --drop-path 0.0 --weight-decay 0.1 
