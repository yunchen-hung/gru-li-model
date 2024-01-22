# srun --time=10:00:00 --mem=16000 --cpus-per-task=1 python -u main.py --exp CondEM --setup setup_recallquestion.json
# python run_cluster.py --exp CondEM --setup setup_recallquestion_pretrain.json -train --time 16
# srun --time=2:00:00 --mem=16000 --cpus-per-task=1 python -u main.py --exp CondEM --setup setup_recallquestion.json

for i in {1..9}
do
/home/ml8736/memory-replay/experiments/RL/figures/cogsci/ValueMemoryGRU/setup_gru_negmementreg_gamma08-16    python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma0$i.json --exp_file cogsci
done

python run_cluster.py --exp RL --setup setup_gru_negmementreg.json --exp_file cogsci
python run_cluster.py --exp RL --setup setup_gru_negmementreg_gamma.json --exp_file cogsci
