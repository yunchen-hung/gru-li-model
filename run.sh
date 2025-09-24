gamma=('0' '02' '04' '06' '08' '10')
noise=('0' '02' '04' '06' '08' '1')

for n in "${noise[@]}"
do
    for g in "${gamma[@]}"
    do
        python run_cluster.py --exp VaryAllSeq8 --cpus_per_task 4 --setup setup_gamma${g}_noise${n}.json --time 11 -train
        # python run_cluster.py --exp VaryAllSeq8 --cpus_per_task 1 --setup setup_gamma${g}_noise${n}.json --time 1 --exp_file time_invariant
    done
done