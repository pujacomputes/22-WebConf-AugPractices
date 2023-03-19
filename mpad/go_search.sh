#!/bin/bash
for seed in 0 1 
do
    for aug_1 in n s e
    do 
        for aug_2 in s n e
        do
        time python run_SimSiam.py --dataset subjectivity \
            --path-to-dataset ../datasets/subjectivity/unique/0.txt \
            --path-to-embeddings ../GoogleNews-vectors-negative300.bin \
            --epochs 25 \
            --no-cuda True \
            --optim ADAM \
            --lr 0.001 \
            --weight_decay 1e-4 \
            --scheduler COSINE \
            --hidden 64 \
            --penultimate 64 \
            --bottleneck 30 \
            --message-passing-layers 2 \
            --window-size 2 \
            --directed True \
            --use-master-node True \
            --normalize True \
            --dropout 0.5 \
            --batch-size 128 \
            --patience 20 \
            --use_nlp_aug False \
            --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
            --rand_aug_1 $aug_1 \
            --rand_aug_2 $aug_2 \
            --rand_node_drop 0.1 \
            --rand_edge_perturb 0.1 \
            --rand_subgraph_drop 0.1 \
            --seed $seed
        done
    done
done


for seed in 0 1 
do
    for aug_1 in n s e
    do 
        for aug_2 in n s e
        do
        time python run_SimSiam.py --dataset subjectivity \
            --path-to-dataset ../datasets/subjectivity/unique/0.txt \
            --path-to-embeddings ../GoogleNews-vectors-negative300.bin \
            --epochs 25 \
            --no-cuda True \
            --optim ADAM \
            --lr 0.001 \
            --weight_decay 1e-4 \
            --scheduler COSINE \
            --hidden 64 \
            --penultimate 64 \
            --bottleneck 30 \
            --message-passing-layers 2 \
            --window-size 2 \
            --directed True \
            --use-master-node True \
            --normalize True \
            --dropout 0.5 \
            --batch-size 128 \
            --patience 20 \
            --use_nlp_aug False \
            --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
            --rand_aug_1 $aug_1 \
            --rand_aug_2 $aug_2 \
            --rand_node_drop 0.05 \
            --rand_edge_perturb 0.05 \
            --rand_subgraph_drop 0.05 \
            --seed $seed
        done
    done
done