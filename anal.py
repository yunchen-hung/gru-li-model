import json
from pathlib import Path
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import RidgeClassifier

from tasks import ConditionalQuestionAnswer
from analysis.decoding import DMAnswerDecoder

from utils import load_dict, savefig



def main():
    trial_num = 5000
    fig_path = Path("./figures_qa")
    env = ConditionalQuestionAnswer(num_features=4, feature_dim=2, sequence_len=4, include_question_during_encode=True)
    timestep_each_phase = 4

    memory_sequences = []
    answers = []
    memory_sequences_int = []

    for _ in range(trial_num):
        env.reset()
        trial_data = env.get_trial_data()
        memory_sequences.append(trial_data["memory_sequence"])
        answers.append(trial_data["correct_answer"])
        memory_sequences_int.append(trial_data["memory_sequence_int"])

    memory_sequences = np.array(memory_sequences)
    answers = np.array(answers)
    memory_sequences_int = np.array(memory_sequences_int)

    memory_sequences_features = np.zeros((trial_num, timestep_each_phase, env.num_features*2))
    for i in range(trial_num):
        for t in range(timestep_each_phase):
            for j in range(env.num_features):
                memory_sequences_features[i, t, j*2+memory_sequences[i, t, j]] = 1
        if i == 0:
            print(memory_sequences[i])
            print(memory_sequences_features[i])

    memory_sequences_features_sum = np.cumsum(memory_sequences_features, axis=1)
    # print(memory_sequences_features_sum[:5])
    memory_sequences_onehot = np.eye(2**env.num_features)[memory_sequences_int]
    memory_sequences_sum_onehot = np.cumsum(memory_sequences_onehot, axis=1)
    print(memory_sequences_sum_onehot.shape, memory_sequences_features_sum.shape)


    ridge_decoder = RidgeClassifier()
    decoder = DMAnswerDecoder(decoder=ridge_decoder)

    decoder.fit(memory_sequences_features_sum.transpose(1, 0, 2), answers)
    final_answer_decode_results = decoder.results
    decoder.visualize(save_path=fig_path/"decode_answer"/"raw_item_features", save_name="all_enc", xlabel="time in encoding phase", figsize=(4, 3.3))

    num_features = env.unwrapped.num_features
    num_questions = num_features * (num_features - 1)
    all_possible_answers = np.zeros((trial_num, num_questions))
    cnt_trial = 0
    for k in range(trial_num):
        mem_seq = memory_sequences[k]
        cnt = 0
        for i in range(num_features):
            for j in range(num_features):
                if i == j:
                    continue
                matched_trial = np.where(mem_seq[:, i] == 1)[0]
                answer = np.sum(mem_seq[matched_trial, j]) % 2
                all_possible_answers[cnt_trial, cnt] = answer
                cnt += 1
        cnt_trial += 1

    plt.figure(figsize=(1.0*timestep_each_phase, 3.3), dpi=180)
    for i in range(num_questions):
        decoder.fit(memory_sequences_features_sum.transpose(1, 0, 2), all_possible_answers[:, i])
        plt.plot(np.arange(1, timestep_each_phase+1), decoder.results, color="tab:blue")
    plt.plot(np.arange(1, final_answer_decode_results.shape[0]+1), final_answer_decode_results, color="tab:orange", label="final answer")
    plt.legend()
    plt.xlabel("time in encoding phase")
    plt.ylabel("answer decoding accuracy")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig(fig_path/"decode_answer"/"raw_item_features", "all_ans_all_enc")


    ridge_decoder = RidgeClassifier()
    decoder = DMAnswerDecoder(decoder=ridge_decoder)

    decoder.fit(memory_sequences_sum_onehot.transpose(1, 0, 2), answers)
    final_answer_decode_results = decoder.results
    decoder.visualize(save_path=fig_path/"decode_answer"/"raw_item_onehot", save_name="all_enc", xlabel="time in encoding phase", figsize=(4, 3.3))

    num_features = env.unwrapped.num_features
    num_questions = num_features * (num_features - 1)
    all_possible_answers = np.zeros((trial_num, num_questions))
    cnt_trial = 0
    for k in range(trial_num):
        mem_seq = memory_sequences[k]
        cnt = 0
        for i in range(num_features):
            for j in range(num_features):
                if i == j:
                    continue
                matched_trial = np.where(mem_seq[:, i] == 1)[0]
                answer = np.sum(mem_seq[matched_trial, j]) % 2
                all_possible_answers[cnt_trial, cnt] = answer
                cnt += 1
        cnt_trial += 1

    plt.figure(figsize=(1.0*timestep_each_phase, 3.3), dpi=180)
    for i in range(num_questions):
        decoder.fit(memory_sequences_sum_onehot.transpose(1, 0, 2), all_possible_answers[:, i])
        plt.plot(np.arange(1, timestep_each_phase+1), decoder.results, color="tab:blue")
    plt.plot(np.arange(1, final_answer_decode_results.shape[0]+1), final_answer_decode_results, color="tab:orange", label="final answer")
    plt.legend()
    plt.xlabel("time in encoding phase")
    plt.ylabel("answer decoding accuracy")
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    savefig(fig_path/"decode_answer"/"raw_item_onehot", "all_ans_all_enc")


if __name__ == "__main__":
    main()