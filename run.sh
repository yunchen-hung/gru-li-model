# # squeue -u <username> -h -t pending,running -r -O "state" | uniq -c

# for i in {1..9}
# do
# python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma0$i.json --time 1
# done

# python run_cluster.py --exp RL --setup setup_gru_negmementreg.json --time 1
# python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma.json --time 1

# python run_cluster.py --exp RL --setup setup_gru_negmementreg.json --exp_file cogsci
# python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma.json --exp_file cogsci

# noise=('0' '02' '04' '06' '08' '1')
# # noise=('0' '1')
# seqlen=('8' '16')

# for s in "${seqlen[@]}"
# do
#     for n in "${noise[@]}"
#     do
#         python run_cluster.py --exp RL.Noise.NBack --setup setup_seq${s}_noise${n}.json --time 1
#     done
# done



# gamma=('0' '01' '02' '03' '04' '05' '06' '07' '08' '09' '1')
gamma=('0' '02' '04' '06' '08' '10')
eta=('0005' '001' '002' '004')

for g in "${gamma[@]}"
do
    for e in "${eta[@]}"
    do
        python run_cluster.py --exp VaryGamma --cpus_per_task 4 --setup setup_eta${e}_gamma${g}.json --time 11 -train
        python run_cluster.py --exp VaryGamma --cpus_per_task 4 --setup setup_pretrain_eta${e}_gamma${g}.json --time 11 -train
    done
done


# noise=('0' '02' '04' '06' '08' '1')
# # noise=('0' '1')
# seqlen=('8' '16')

# for s in "${seqlen[@]}"
# do
#     for n in "${noise[@]}"
#     do
#         python run_cluster.py --exp FreeRecall.VaryNoise --cpus_per_task 8 --setup setup_seq${s}_noise${n}.json --time 20 -train
#     done
# done




# for s in "${seqlen[@]}"
# do
#     for n in "${noise[@]}"
#     do
#         python run_cluster.py --exp RL.Noise.NBack --setup setup_seq${s}_noise${n}.json --time 1
#     done
# done


# nback_dir="./experiments/RL/NBack/VarySeq/setups/"
# if [ ! -d $nback_dir ]; then
#   echo "Directory $nback_dir does not exist"
#   exit 1
# fi

# for file in "$nback_dir"/*; do
#   if [ -f "$file" ]; then
#     name=${file##*/}
#     python run_cluster.py --exp RL.NBack.VarySeq --setup $name --time 10
#   fi
# done


# python run_cluster.py --exp CondQA.Sup --setup setup_encq_last_nomem.json -train --time 10

