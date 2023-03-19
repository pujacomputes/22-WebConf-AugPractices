mode=rand
for dataset in REDDIT-BINARY IMDB-BINARY
	do
	for seed in  1 2 3 4
	do
		python inductivebias_svm.py --mode=$mode --dataset=$dataset --batch_size=32
	done
done

num_layers=4
mode=rand
for dataset in REDDIT-BINARY IMDB-BINARY
	do
	for seed in 0 1 2 3 4
	do
		python inductivebias_svm.py --mode=$mode --dataset=$dataset --batch_size=32 --num_layers $num_layers
	done
done

num_layers=3
mode=rand
for dataset in  REDDIT-BINARY IMDB-BINARY
	do
	for seed in 0 1 2 3 4
	do
		python inductivebias_svm.py --mode=$mode --dataset=$dataset --batch_size=32 --num_layers $num_layers
	done
done

num_layers=2
mode=rand
for dataset in REDDIT-BINARY IMDB-BINARY
	do
	for seed in 0 1 2 3 4
	do
		python inductivebias_svm.py --mode=$mode --dataset=$dataset --batch_size=32 --num_layers $num_layers
	done
done
