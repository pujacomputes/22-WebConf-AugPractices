#!/bin/bash -ex
for dataset in MUTAG PROTEINS DD
do 
	for seed in 0 1 2 3 4
	do 
		CUDA_VISIBLE_DEVICES=1 python gsimclr.py --DS $dataset --lr 0.001 --num-gc-layers 5 --aug random2 --seed $seed --epochs 20 --hidden-dim 32
	done
done 
