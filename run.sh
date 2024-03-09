# srun --time=10:00:00 --mem=16000 --cpus-per-task=1 python -u main.py --exp CondEM --setup setup_recallquestion.json
# python run_cluster.py --exp CondEM --setup setup_recallquestion_pretrain.json -train --time 16
# srun --time=2:00:00 --mem=16000 --cpus-per-task=1 python -u main.py --exp CondEM --setup setup_recallquestion.json

# for i in {1..9}
# do
# python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma0$i.json --exp_file cogsci
# done

# python run_cluster.py --exp RL --setup setup_gru_negmementreg.json --exp_file cogsci
# python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma.json --exp_file cogsci


noise=('0' '02' '04' '06' '08' '1')
seqlen=('4' '12' '16')

for s in "${seqlen[@]}"
do
    for n in "${noise[@]}"
    do
        for g in {0..9..3}
        do
        python run_cluster.py --exp RL.Noise.Seq${s} --setup setup_seq${s}_noise${n}_gamma0${g}.json --time 10
        done
    done
done
