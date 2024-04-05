# srun --time=10:00:00 --mem=16000 --cpus-per-task=1 python -u main.py --exp CondEM --setup setup_recallquestion.json
# python run_cluster.py --exp CondEM --setup setup_recallquestion_pretrain.json -train --time 16
# srun --time=2:00:00 --mem=16000 --cpus-per-task=1 python -u main.py --exp CondEM --setup setup_recallquestion.json

# for i in {1..9}
# do
# python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma0$i.json --exp_file cogsci
# done

# python run_cluster.py --exp RL --setup setup_gru_negmementreg.json --exp_file cogsci
# python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma.json --exp_file cogsci


# noise=('0' '02' '04' '06' '08' '1')
# seqlen=('4' '8' '12' '16')

# for s in "${seqlen[@]}"
# do
#     for n in "${noise[@]}"
#     do
#         python run_cluster.py --exp RL.Noise.NBack --setup setup_seq${s}_noise${n}.json --time 10
#     done
# done

python run_cluster.py --exp CondQA --setup setup_encq.json --time 15 -train
python run_cluster.py --exp CondQA --setup setup_recq.json --time 15 -train

# python run_cluster.py --exp CondEM --setup setup_encq_pretrain_gamma09.json --time 15 -train
# python run_cluster.py --exp CondEM --setup setup_encq_pretrain.json --time 15 -train
# python run_cluster.py --exp CondEM --setup setup_recq_pretrain_gamma09.json --time 15 -train
# python run_cluster.py --exp CondEM --setup setup_recq_pretrain.json --time 15 -train
# python run_cluster.py --exp CondEM --setup setup_encq_pretrain_gamma09.json
# python run_cluster.py --exp CondEM --setup setup_encq_pretrain.json
# python run_cluster.py --exp CondEM --setup setup_recq_pretrain_gamma09.json
# python run_cluster.py --exp CondEM --setup setup_recq_pretrain.json


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

