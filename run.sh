# # squeue -u <username> -h -t pending,running -r -O "state" | uniq -c


# gamma=('0' '01' '02' '03' '04' '05' '06' '07' '08' '09' '10')
gamma=('0' '02' '04' '06' '08' '10')
# gamma=('08' '085' '09' '095' '099' '0999' '09999')
# gamma=('0' '01' '02' '03' '04')
eta=('0005' '001' '002' '004')

# for g in "${gamma[@]}"
# do
#     for e in "${eta[@]}"
#     do
#         # python run_cluster.py --exp VaryGammaSeq12 --cpus_per_task 4 --setup setup_eta${e}_gamma${g}.json --time 16 -train
#         # python run_cluster.py --exp VaryGammaSeq12 --cpus_per_task 1 --setup setup_eta${e}_gamma${g}.json --time 2
#         scancel -n VaryGammaSeq12.setup_eta${e}_gamma${g}
#     done
# done

# for g in "${gamma[@]}"
# do
#     # python run_cluster.py --exp VaryGamma2 --cpus_per_task 4 --setup setup_gamma${g}.json --time 11 -train
#     python run_cluster.py --exp VaryGamma2 --cpus_per_task 4 --setup setup_gamma${g}.json --time 11 -train
# done


noise=('0' '02' '04' '06' '08' '1')
# noise=('06' '08' '1')
# # noise=('0' '1')
# seqlen=('8' '16')

# for n in "${noise[@]}"
# do
#     # python run_cluster.py --exp VaryNoiseSeq12Decay --cpus_per_task 4 --setup setup_noise${n}.json --time 20 -train
#     python run_cluster.py --exp VaryNoiseSeq12Decay --cpus_per_task 1 --setup setup_noise${n}.json --time 2
# done

# for n in "${noise[@]}"
# do
#     python run_cluster.py --exp VaryNoise --cpus_per_task 4 --setup setup_seq8_noise${n}.json --time 10 -train
#     python run_cluster.py --exp VaryNoise --cpus_per_task 4 --setup setup_pretrain_seq8_noise${n}.json --time 10 -train
# done

# for n in "${noise[@]}"
# do
#     python run_cluster.py --exp VaryNoise --cpus_per_task 4 --setup setup_seq16_noise${n}.json --time 20 -train
#     python run_cluster.py --exp VaryNoise --cpus_per_task 4 --setup setup_pretrain_seq16_noise${n}.json --time 20 -train
# done



# for n in "${noise[@]}"
# do
#     for g in "${gamma[@]}"
#     do
#         python run_cluster.py --exp VaryAllSeq12 --cpus_per_task 8 --setup setup_gamma${g}_noise${n}.json --time 24 -train
#     done
# done

# for n in "${noise[@]}"
# do
#     # python run_cluster.py --exp VaryAllSeq12NoNoise --cpus_per_task 8 --setup setup_gamma10_noise${n}.json --time 20 -train
#     # python run_cluster.py --exp VaryAllSeq8NoNoise --cpus_per_task 4 --setup setup_gamma10_noise${n}.json --time 11 -train
#     python run_cluster.py --exp VaryAllSeq8 --cpus_per_task 1 --setup setup_gamma10_noise${n}.json --time 1 --exp_file perturbation
# done

# for g in "${gamma[@]}"
# do
#     # python run_cluster.py --exp VaryAllSeq12NoNoise --cpus_per_task 8 --setup setup_gamma${g}_noise1.json --time 20 -train
#     # python run_cluster.py --exp VaryAllSeq8 --cpus_per_task 4 --setup setup_gamma${g}_noise1.json --time 11 -train
#     python run_cluster.py --exp VaryAllSeq8NoNoise --cpus_per_task 1 --setup setup_gamma${g}_noise1.json --time 1 --exp_file perturbation
# done

# for n in "${noise[@]}"
# do
#     for g in "${gamma[@]}"
#     do
#         # python run_cluster.py --exp VaryAllSeq8LargeNoise --cpus_per_task 4 --setup setup_gamma${g}_noise${n}.json --time 11 -train
#         python run_cluster.py --exp VaryAllSeq8LargeNoise --cpus_per_task 1 --setup setup_gamma${g}_noise${n}.json --time 1 --exp_file time_invariant
#         # --exp_file item_invariant
#     done
# done


hidden_dim=('16' '32' '64' '128' '256')

# for d in "${hidden_dim[@]}"
# do
#     # python run_cluster.py --exp VaryHiddenDim --cpus_per_task 8 --setup setup_dim${d}.json --time 20 -train
#     python run_cluster.py --exp VaryHiddenDim --cpus_per_task 1 --setup setup_dim${d}.json --time 2
# done


# wm_noise=('0' '01' '02' '04' '06' '08')
wm_noise=('0' '001' '004' '009' '016' '025')

# for w in "${wm_noise[@]}"
# do
#     python run_cluster.py --exp VaryWMNoise --cpus_per_task 8 --setup setup_wmnoise${w}.json --time 20 -train
#     # python run_cluster.py --exp VaryWMNoise --cpus_per_task 1 --setup setup_wmnoise${w}.json --time 2
# done


gamma=('0' '1')
wm_noise=('0' '005' '01')
# seq_len=('4' '8' '12' '16')
# train_time=(6 11 23 47)
seq_len=('4' '8' '12')
train_time=(6 11 23)
# seq_len=('16')
# train_time=(24)

for i in "${!seq_len[@]}"
do
    s="${seq_len[i]}"
    t="${train_time[i]}"
    for w in "${wm_noise[@]}"
    do
        for g in "${gamma[@]}"
        do
            # python run_cluster.py --exp VarySeqLenNoise --cpus_per_task 8 --setup setup_seq${s}_gate_gamma${g}_wmnoise${w}.json --time ${t} -train
            python run_cluster.py --exp VarySeqLenNoise --cpus_per_task 1 --setup setup_seq${s}_gamma${g}_wmnoise${w}.json --time 1 --mem 16
        done
    done
done

