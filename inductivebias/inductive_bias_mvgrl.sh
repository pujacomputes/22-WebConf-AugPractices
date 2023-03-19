for dataset in MUTAG PROTEINS NCI1 DD
 	do
	for mode in e2e rand bn
		do 
		python inductive_bias.py --mode=$mode --dataset=$dataset --batch_size=32
	done
done
	num_nodes=50
for mode in e2e rand bn
do 
	python inductive_bias_mvgrl.py --mode=$mode --dataset=$dataset --num_nodes=$num_nodes
done
dataset=PROTEINS
num_nodes=60
for mode in e2e rand bn
do 
	python inductive_bias_mvgrl.py --mode=$mode --dataset=$dataset --num_nodes=$num_nodes
done
dataset=NCI1
num_nodes=100
for mode in e2e rand bn
do 
	python inductive_bias_mvgrl.py --mode=$mode --dataset=$dataset --num_nodes=$num_nodes
done
dataset=DD
num_nodes=400
for mode in e2e rand bn
do 
	python inductive_bias_mvgrl.py --mode=$mode --dataset=$dataset --num_nodes=$num_nodes
done
