#!/bin/bash
window=4
# for seed in 1 2 3
# do 
# time python run_RAND.py --dataset subjectivity \
#     --path-to-dataset ../datasets/subjectivity/unique/0.txt \
#     --path-to-embeddings ../GoogleNews-vectors-negative300.bin \
#     --epochs 25 \
#     --no-cuda True \
#     --optim ADAM \
#     --lr 0.001 \
#     --weight_decay 1e-4 \
#     --scheduler COSINE \
#     --hidden 64 \
#     --penultimate 64 \
#     --bottleneck 30 \
#     --message-passing-layers 2 \
#     --window-size $window \
#     --directed True \
#     --use-master-node True \
#     --normalize True \
#     --dropout 0.5 \
#     --batch-size 128 \
#     --patience 20 \
#     --use_nlp_aug False \
#     --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
#     --rand_aug_1 e \
#     --rand_aug_2 s \
#     --rand_node_drop 0.1 \
#     --rand_edge_perturb 0.1 \
#     --rand_subgraph_drop 0.1 \
#     --seed $seed
# done


for seed in 0 1 2
do
for mp_type in gcn sage gin
do
        time python run_BYOL_v2.py --dataset subjectivity \
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
                --rand_aug_1 s \
                --rand_aug_2 s \
                --rand_node_drop 0.1 \
                --rand_edge_perturb 0.1 \
                --rand_subgraph_drop 0.1 \
                --mp_type $mp_type \
                --seed $seed

        time python run_BYOL_v2.py --dataset subjectivity \
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
        --use_nlp_aug True \
        --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
        --rand_aug_1 s \
        --rand_aug_2 n \
        --rand_node_drop 0.1 \
        --rand_edge_perturb 0.1 \
        --rand_subgraph_drop 0.1 \
        --mp_type $mp_type \
        --seed $seed
        
        time python run_BYOL_v2.py --dataset subjectivity \
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
        --use_nlp_aug True \
        --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
        --rand_aug_1 X \
        --rand_aug_2 X \
        --rand_node_drop -1 \
        --rand_edge_perturb -1 \
        --rand_subgraph_drop -1 \
        --seed 0

        
        done
done 

for seed in 0 1 2
        do
for mp_type in gcn sage gin
do
        time python run_SimSiam_v2.py --dataset subjectivity \
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
                --rand_aug_1 s \
                --rand_aug_2 s \
                --rand_node_drop 0.1 \
                --rand_edge_perturb 0.1 \
                --rand_subgraph_drop 0.1 \
                --mp_type $mp_type \
                --seed $seed

        time python run_SimSiam_v2.py --dataset subjectivity \
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
        --use_nlp_aug True \
        --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
        --rand_aug_1 s \
        --rand_aug_2 n \
        --rand_node_drop 0.1 \
        --rand_edge_perturb 0.1 \
        --rand_subgraph_drop 0.1 \
        --mp_type $mp_type \
        --seed $seed
        
        time python run_SimSiam_v2.py --dataset subjectivity \
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
        --use_nlp_aug True \
        --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
        --rand_aug_1 X \
        --rand_aug_2 X \
        --rand_node_drop -1 \
        --rand_edge_perturb -1 \
        --rand_subgraph_drop -1 \
        --seed 0
        done
done 


# time python run_SimSiam.py --dataset subjectivity \
#     --path-to-dataset ../datasets/subjectivity/unique/0.txt \
#     --path-to-embeddings ../GoogleNews-vectors-negative300.bin \
#     --epochs 25 \
#     --no-cuda True \
#     --optim ADAM \
#     --lr 0.001 \
#     --weight_decay 1e-4 \
#     --scheduler COSINE \
#     --hidden 64 \
#     --penultimate 64 \
#     --bottleneck 30 \
#     --message-passing-layers 2 \
#     --window-size 2 \
#     --directed True \
#     --use-master-node True \
#     --normalize True \
#     --dropout 0.5 \
#     --batch-size 128 \
#     --patience 20 \
#     --use_nlp_aug True \
#     --consolidated_file ../datasets/subjectivity/unique/consolidated.txt \
#     --rand_aug_1 X \
#     --rand_aug_2 X \
#     --rand_node_drop -1 \
#     --rand_edge_perturb -1 \
#     --rand_subgraph_drop -1 \
#     --seed 0
