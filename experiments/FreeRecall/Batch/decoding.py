import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, LogisticRegression

from utils import savefig
from analysis.decoding import ItemIdentityDecoder, ItemIndexDecoder



def run(data_all, model_all, env, paths, exp_name):
    plt.rcParams['font.size'] = 16

    env = env[0]

    for run_name, data in data_all.items():
        run_name_without_num = run_name.split("-")[0]
        # fig_path = paths["fig"]/run_name
        run_num = run_name.split("-")[-1]
        fig_path = paths["fig"]/run_name_without_num/run_num
        fig_path.mkdir(parents=True, exist_ok=True)
        print()
        print(run_name)

        data = data[0]

        model = model_all[run_name]
        if hasattr(model, "step_for_each_timestep"):
            step_for_each_timestep = model.step_for_each_timestep
            timestep_each_phase = step_for_each_timestep * env.memory_num
        else:
            step_for_each_timestep = 1
            timestep_each_phase = env.memory_num
        # timestep_each_phase = env.memory_num

        # get recorded data and outputs of the model
        readouts = data['readouts']
        actions = data['actions']
        rewards = data['rewards']

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)

        # convert data to numpy array
        memory_contexts = np.array(data['trial_data'])     # ground truth of memory for each trial
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        actions = np.array(actions).squeeze(-1)
        # print(actions.shape)
        actions = actions.reshape(-1, actions.shape[-1])        # (trials, timesteps per trial)
        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])        # (trials, timesteps per trial)

        print(memory_contexts.shape, actions.shape, rewards.shape)

        if "ValueMemory" in readouts[0] and "similarity" in readouts[0]["ValueMemory"]:
            has_memory = True
        else:
            has_memory = False

        

        """ ridge classification """
        """ decode item identity """
        retrieved_memories = []
        for i in range(all_context_num):
            retrieved_memory = readouts[i]["ValueMemory"]["similarity"].squeeze()
            retrieved_memory = np.argmax(retrieved_memory, axis=-1)
            retrieved_memories.append(retrieved_memory)
        retrieved_memories = np.stack(retrieved_memories)

        c_memorizing = np.stack([readouts[i]['state'][:timestep_each_phase].squeeze() for i in range(all_context_num)])   # context_num * time * state_dim
        c_recalling = np.stack([readouts[i]['state'][-timestep_each_phase:].squeeze() for i in range(all_context_num)])
        memory_sequence = np.stack([memory_contexts[i] for i in range(all_context_num)]) - 1    # context_num * time
        memory_sequence_onehot = np.eye(env.vocabulary_num)[memory_sequence]        # context_num * time * vocabulary_num
        actions_onehot = np.eye(env.vocabulary_num+1)[actions]                        # context_num * time * vocabulary_num

        # Ridge
        ridge_decoder = RidgeClassifier()
        ridge = ItemIdentityDecoder(decoder=ridge_decoder)
        ridge_encoding_res, ridge_encoding_stat_res = ridge.fit(c_memorizing.transpose(1, 0, 2), memory_sequence_onehot.transpose(1, 0, 2))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")
        # np.save(fig_path/"ridge_encoding.npy", ridge_encoding_res)
        # np.save(fig_path/"ridge_encoding_stat.npy", list(ridge_encoding_stat_res.values()))

        _, _ = ridge.fit_no_crossval(c_memorizing.transpose(1, 0, 2), memory_sequence_onehot.transpose(1, 0, 2))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_no_crossval", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")
        
        _, _ = ridge.fit(c_memorizing.transpose(1, 0, 2), memory_sequence.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc_onedim", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")


        ridge_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(env.memory_num):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    ridge_mask[i][t] = 1
        ridge_recall_res, ridge_recall_stat_res = ridge.fit(c_recalling.transpose(1, 0, 2), actions_onehot[:, -timestep_each_phase:].transpose(1, 0, 2), 
                                                            ridge_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")
        # np.save(fig_path/"ridge_recall.npy", ridge_recall_res)
        # np.save(fig_path/"ridge_recall_stat.npy", list(ridge_recall_stat_res.values()))
        _, _ = ridge.fit_no_crossval(c_recalling.transpose(1, 0, 2), actions_onehot[:, -timestep_each_phase:].transpose(1, 0, 2),
                                    ridge_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_no_crossval", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")
        
        _, _ = ridge.fit(c_recalling.transpose(1, 0, 2), actions[:, -timestep_each_phase:].transpose(1, 0),
                                    ridge_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec_onedim", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")


        # logistic regression
        logistic_decoder = LogisticRegression(max_iter=1000)
        logistic = ItemIdentityDecoder(decoder=logistic_decoder)
        logistic_encoding_res, logistic_encoding_stat_res = logistic.fit(c_memorizing.transpose(1, 0, 2), memory_sequence.transpose(1, 0))
        logistic.visualize_by_memory(save_path=fig_path/"logistic", save_name="c_enc", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")
        _, _ = logistic.fit_no_crossval(c_memorizing.transpose(1, 0, 2), memory_sequence.transpose(1, 0))
        logistic.visualize_by_memory(save_path=fig_path/"logistic", save_name="c_enc_no_crossval", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")
                    
        logistic_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(env.memory_num):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    logistic_mask[i][t] = 1
        logistic_recall_res, logistic_recall_stat_res = logistic.fit(c_recalling.transpose(1, 0, 2), actions[:, -timestep_each_phase:].transpose(1, 0),
                                                            logistic_mask.transpose(1, 0))
        logistic.visualize_by_memory(save_path=fig_path/"logistic", save_name="c_rec", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")
        _, _ = logistic.fit_no_crossval(c_recalling.transpose(1, 0, 2), actions[:, -timestep_each_phase:].transpose(1, 0),
                                    logistic_mask.transpose(1, 0))
        logistic.visualize_by_memory(save_path=fig_path/"logistic", save_name="c_rec_no_crossval", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")


        """ decode item index """
        # encoding_index = np.repeat(np.arange(env.memory_num).reshape(1, -1), all_context_num, axis=0)

        # recall_index = np.zeros_like(actions[:, -timestep_each_phase:])
        # index_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        # for i in range(all_context_num):
        #     for t in range(env.memory_num):
        #         if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
        #             index_mask[i][t] = 1
        #             recall_index[i][t] = np.where(memory_contexts[i] == actions[i][-timestep_each_phase+t])[0][0]

        # # Ridge
        # ridge_decoder = RidgeClassifier()
        # ridge = ItemIndexDecoder(decoder=ridge_decoder)
        # ridge_encoding_res, index_encoding_acc, index_encoding_r2 = ridge.fit(c_memorizing, encoding_index)
        # ridge.visualize(save_path=fig_path/"ridge_index", save_name="c_enc", xlabel="time in encoding phase")
        # np.save(fig_path/"ridge_encoding_index.npy", ridge_encoding_res)

        # ridge_recall_res, index_recall_acc, index_recall_r2 = ridge.fit(c_recalling, recall_index, index_mask)
        # ridge.visualize(save_path=fig_path/"ridge_index", save_name="c_rec", xlabel="time in recall phase")
        # np.save(fig_path/"ridge_recall_index.npy", ridge_recall_res)

        # ridge_classifier_stat = {
        #     "item_enc_acc": ridge_encoding_stat_res["acc"],
        #     "item_enc_r2": ridge_encoding_stat_res["r2"],
        #     "item_enc_acc_last": ridge_encoding_stat_res["acc_last"],
        #     "item_enc_r2_last": ridge_encoding_stat_res["r2_last"],
        #     "item_rec_acc": ridge_recall_stat_res["acc"],
        #     "item_rec_r2": ridge_recall_stat_res["r2"],
        #     "item_rec_acc_last": ridge_recall_stat_res["acc_last"],
        #     "item_rec_r2_last": ridge_recall_stat_res["r2_last"],
        #     "index_enc_acc": index_encoding_acc,
        #     "index_enc_r2": index_encoding_r2,
        #     "index_rec_acc": index_recall_acc,
        #     "index_rec_r2": index_recall_r2
        # }
        # with open(fig_path/"ridge_classifier_stat.pkl", "wb") as f:
        #     pickle.dump(ridge_classifier_stat, f)
