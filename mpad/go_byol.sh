window=4
seed=1
nocuda=True
time python run_BYOL.py --dataset subjectivity \
        --path-to-dataset ../datasets/subjectivity/unique/0.txt \
        --path-to-embeddings ../GoogleNews-vectors-negative300.bin \
        --epochs 25 \
        --no-cuda $nocuda \
        --optim ADAM \
        --lr 0.001 \
        --weight_decay 1e-4 \
        --scheduler COSINE \
        --hidden 64 \
        --penultimate 64 \
        --bottleneck 30 \
        --message-passing-layers 2 \
        --window-size $window \
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
        --seed $seed