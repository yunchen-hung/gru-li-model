import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics.pairwise as skp
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, adjusted_mutual_info_score

from utils import savefig
from analysis.decomposition import PCA
from analysis.decoding import PCSelectivity, ItemIdentityDecoder, ItemIndexDecoder, Regressor, Classifier
from analysis.behavior import RecallProbability, RecallProbabilityInTime, TemporalFactor



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
        
        # print ground truths, actions and rewards for 5 trials
        print("accuracy: {}".format(data['accuracy']))
        for i in range(5):
            if has_memory:
                print("context {}, gt: {}, action: {}, retrieved memory: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                np.argmax(readouts[i]["ValueMemory"]["similarity"].squeeze(), axis=1)+1, rewards[i][env.memory_num:]))
            else:
                print("context {}, gt: {}, action: {}, rewards: {}".format(i, memory_contexts[i], actions[i][env.memory_num:], 
                rewards[i][env.memory_num:]))

        """ similarity of states """
        similarities = []
        for i in range(context_num):
            states = readouts[i]["state"].squeeze()
            similarity = skp.cosine_similarity(states, states)
            similarities.append(similarity)
        similarities = np.stack(similarities)
        similarity = np.mean(similarities, axis=0)

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(similarity[timestep_each_phase:timestep_each_phase*2, :timestep_each_phase], cmap="Blues")
        plt.colorbar(label="cosine similarity\nbetween hidden states")
        plt.xlabel("time in encoding phase")
        plt.ylabel("time in recall phase")
        # plt.title("encoding-recalling state similarity")
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "encode_recall")

        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(similarity[:timestep_each_phase, :timestep_each_phase], cmap="Blues")
        plt.colorbar(label="cosine similarity\nbetween hidden states")
        plt.xlabel("time in encoding phase")
        plt.ylabel("time in encoding phase")
        # plt.title("encoding state similarity")
        plt.tight_layout()
        savefig(fig_path/"state_similarity", "encode_encode")

        np.save(fig_path/"state_similarity"/"state_similarity.npy", similarity)

        """ memory gate """
        # if "mem_gate_recall" in readouts[0]:
        #     plt.figure(figsize=(4, 3), dpi=180)
        #     for i in range(context_num):
        #         em_gates = readouts[i]['mem_gate_recall']
        #         plt.plot(np.mean(em_gates.squeeze(1), axis=-1)[:timestep_each_phase], label="context {}".format(i))
        #     ax = plt.gca()
        #     ax.spines['top'].set_visible(False)
        #     ax.spines['right'].set_visible(False)
        #     plt.xlabel("time of recall phase")
        #     plt.ylabel("memory gate")
        #     plt.tight_layout()
        #     savefig(fig_path, "em_gate_recall")

        """ recall probability (output) (CRP curve) """
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        # plot CRP curve
        recall_probability.visualize_all_time(fig_path/"recall_prob", format="svg")
        recall_probability.visualize(fig_path/"recall_prob", format="svg")
        results_all_time = recall_probability.get_results_all_time()
        # write to csv file
        with open(fig_path/"recall_probability.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(results_all_time)

        """ count temporal factor and forward asymmetry """
        recall_probability = RecallProbability()
        recall_probability.fit(memory_contexts, actions[:, -timestep_each_phase:])
        forward_asymmetry = recall_probability.forward_asymmetry
        temporal_factor = TemporalFactor()
        temp_fact = temporal_factor.fit(memory_contexts, actions[:, -timestep_each_phase:])
        temp_fact = np.mean(temp_fact)
        print("forward asymmetry:[{},{}]".format(data['accuracy'], forward_asymmetry))
        print("temporal factor:[{},{}]".format(data['accuracy'], temp_fact))
        # write to csv file
        with open(fig_path/"contiguity_effect.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow([data['accuracy'], forward_asymmetry, temp_fact])

        """ recall probability of first timestep (see primacy and recency) """
        recall_probability_in_time = RecallProbabilityInTime()
        recall_probability_in_time.fit(memory_contexts, actions[:, -timestep_each_phase:])
        recall_probability_in_time.visualize(fig_path)

        """ PCA """
        states = []
        for i in range(10):
            states.append(readouts[i]['state'])
        states = np.stack(states).squeeze()
        
        pca = PCA()
        pca.fit(states)
        pca.visualize_state_space(save_path=fig_path/"pca_state_space", end_step=timestep_each_phase, colormap_label="time in\nencoding phase", 
                                file_name="encoding", format="svg")
        pca.visualize_state_space(save_path=fig_path/"pca_state_space", start_step=timestep_each_phase, end_step=timestep_each_phase*2,
                                colormap_label="time in recall phase", file_name="recall", format="svg")

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

        os.makedirs(fig_path/"decoding_data", exist_ok=True)

        # Ridge
        ridge_decoder = RidgeClassifier()
        ridge = ItemIdentityDecoder(decoder=ridge_decoder)
        ridge_encoding_res, ridge_encoding_stat_res = ridge.fit(c_memorizing.transpose(1, 0, 2), memory_sequence.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_enc", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")
        np.save(fig_path/"decoding_data"/"ridge_encoding.npy", ridge_encoding_res)
        # np.save(fig_path/"ridge_encoding_stat.npy", list(ridge_encoding_stat_res.values()))

        ridge_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(env.memory_num):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    ridge_mask[i][t] = 1
        ridge_recall_res, ridge_recall_stat_res = ridge.fit(c_recalling.transpose(1, 0, 2), actions[:, -timestep_each_phase:].transpose(1, 0), ridge_mask.transpose(1, 0))
        ridge.visualize_by_memory(save_path=fig_path/"ridge", save_name="c_rec", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")
        np.save(fig_path/"decoding_data"/"ridge_recall.npy", ridge_recall_res)

        # SVM
        svm = ItemIdentityDecoder()
        svm_encoding_res, svm_encoding_stat_res = svm.fit(c_memorizing.transpose(1, 0, 2), memory_sequence.transpose(1, 0))
        svm.visualize_by_memory(save_path=fig_path/"svm", save_name="c_enc", colormap_label="item position\nin study order",
                                xlabel="time in encoding phase")
        np.save(fig_path/"decoding_data"/"svm_encoding.npy", svm_encoding_res)

        svm_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(env.memory_num):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    svm_mask[i][t] = 1
        svm_recall_res, svm_recall_stat_res = svm.fit(c_recalling.transpose(1, 0, 2), actions[:, -timestep_each_phase:].transpose(1, 0), svm_mask.transpose(1, 0))
        svm.visualize_by_memory(save_path=fig_path/"svm", save_name="c_rec", colormap_label="item position\nin recall order",
                                xlabel="time in recall phase")
        np.save(fig_path/"decoding_data"/"svm_recall.npy", svm_recall_res)




        """ decode item index """
        encoding_index = np.repeat(np.arange(env.memory_num).reshape(1, -1), all_context_num, axis=0)

        recall_index = np.zeros_like(actions[:, -timestep_each_phase:])
        index_mask = np.zeros_like(actions[:, -timestep_each_phase:], dtype=bool)
        for i in range(all_context_num):
            for t in range(env.memory_num):
                if actions[i][-timestep_each_phase+t] in memory_contexts[i]:
                    index_mask[i][t] = 1
                    recall_index[i][t] = np.where(memory_contexts[i] == actions[i][-timestep_each_phase+t])[0][0]

        # Ridge
        ridge_decoder = RidgeClassifier()
        ridge = ItemIndexDecoder(decoder=ridge_decoder)
        ridge_encoding_res, index_encoding_acc, index_encoding_r2 = ridge.fit(c_memorizing, encoding_index)
        ridge.visualize(save_path=fig_path/"ridge_index", save_name="c_enc", xlabel="time in encoding phase")
        np.save(fig_path/"decoding_data"/"ridge_encoding_index.npy", ridge_encoding_res)

        ridge_recall_res, index_recall_acc, index_recall_r2 = ridge.fit(c_recalling, recall_index, index_mask)
        ridge.visualize(save_path=fig_path/"ridge_index", save_name="c_rec", xlabel="time in recall phase")
        np.save(fig_path/"decoding_data"/"ridge_recall_index.npy", ridge_recall_res)

        ridge_classifier_stat = {
            "item_enc_acc": ridge_encoding_stat_res["acc"],
            "item_enc_r2": ridge_encoding_stat_res["r2"],
            "item_enc_acc_last": ridge_encoding_stat_res["acc_last"],
            "item_enc_r2_last": ridge_encoding_stat_res["r2_last"],
            "item_rec_acc": ridge_recall_stat_res["acc"],
            "item_rec_r2": ridge_recall_stat_res["r2"],
            "item_rec_acc_last": ridge_recall_stat_res["acc_last"],
            "item_rec_r2_last": ridge_recall_stat_res["r2_last"],
            "index_enc_acc": index_encoding_acc,
            "index_enc_r2": index_encoding_r2,
            "index_rec_acc": index_recall_acc,
            "index_rec_r2": index_recall_r2
        }
        with open(fig_path/"decoding_data"/"ridge_classifier_stat.pkl", "wb") as f:
            pickle.dump(ridge_classifier_stat, f)


        # SVM
        svm = ItemIndexDecoder()
        svm_encoding_res, index_encoding_acc, index_encoding_r2 = svm.fit(c_memorizing, encoding_index)
        svm.visualize(save_path=fig_path/"svm_index", save_name="c_enc", xlabel="time in encoding phase")
        np.save(fig_path/"decoding_data"/"svm_encoding_index.npy", svm_encoding_res)

        svm_recall_res, index_recall_acc, index_recall_r2 = svm.fit(c_recalling, recall_index, index_mask)
        svm.visualize(save_path=fig_path/"svm_index", save_name="c_rec", xlabel="time in recall phase")
        np.save(fig_path/"decoding_data"/"svm_recall_index.npy", svm_recall_res)

        svm_classifier_stat = {
            "item_enc_acc": svm_encoding_stat_res["acc"],
            "item_enc_r2": svm_encoding_stat_res["r2"],
            "item_enc_acc_last": svm_encoding_stat_res["acc_last"],
            "item_enc_r2_last": svm_encoding_stat_res["r2_last"],
            "item_rec_acc": svm_recall_stat_res["acc"],
            "item_rec_r2": svm_recall_stat_res["r2"],
            "item_rec_acc_last": svm_recall_stat_res["acc_last"],
            "item_rec_r2_last": svm_recall_stat_res["r2_last"],
            "index_enc_acc": index_encoding_acc,
            "index_enc_r2": index_encoding_r2,
            "index_rec_acc": index_recall_acc,
            "index_rec_r2": index_recall_r2
        }
        with open(fig_path/"decoding_data"/"svm_classifier_stat.pkl", "wb") as f:
            pickle.dump(svm_classifier_stat, f)




        """ overall decoding accuracy and r2 """

        def regression(decoder, file_name):
            regressor = Regressor(decoder=decoder)
            r2_identity_enc, ev_identity_enc = regressor.fit(c_memorizing, memory_sequence)
            r2_identity_rec, ev_identity_rec = regressor.fit(c_recalling, actions[:, -timestep_each_phase:], ridge_mask)
            r2_identity_last_enc, ev_identity_last_enc = regressor.fit(c_memorizing[:, 1:], memory_sequence[:, :-1])
            r2_identity_last_rec, ev_identity_last_rec = regressor.fit(c_recalling[:, 1:], actions[:, -timestep_each_phase:-1], ridge_mask[:, :-1])
            r2_index_enc, ev_index_enc = regressor.fit(c_memorizing, encoding_index)
            r2_index_rec, ev_index_rec = regressor.fit(c_recalling, recall_index, index_mask)
            regression_stat = {
                "item_enc_r2": r2_identity_enc,
                "item_enc_ev": ev_identity_enc,
                "item_rec_r2": r2_identity_rec,
                "item_rec_ev": ev_identity_rec,
                "item_enc_r2_last": r2_identity_last_enc,
                "item_enc_ev_last": ev_identity_last_enc,
                "item_rec_r2_last": r2_identity_last_rec,
                "item_rec_ev_last": ev_identity_last_rec,
                "index_enc_r2": r2_index_enc,
                "index_enc_ev": ev_index_enc,
                "index_rec_r2": r2_index_rec,
                "index_rec_ev": ev_index_rec
            }
            with open(fig_path/"decoding_data"/f"{file_name}_regression_stat.pkl", "wb") as f:
                pickle.dump(regression_stat, f)
            
            print()
            print("{} regression".format(file_name))
            print("item identity encoding r2: {}, ev: {}".format(r2_identity_enc, ev_identity_enc))
            print("item identity encoding last r2: {}, ev: {}".format(r2_identity_last_enc, ev_identity_last_enc))
            print("item identity recall r2: {}, ev: {}".format(r2_identity_rec, ev_identity_rec))
            print("item identity recall last r2: {}, ev: {}".format(r2_identity_last_rec, ev_identity_last_rec))
            print("item index encoding r2: {}, ev: {}".format(r2_index_enc, ev_index_enc))
            print("item index recall r2: {}, ev: {}".format(r2_index_rec, ev_index_rec))

        # Ridge
        regression(Ridge(), "ridge")

        # # Lasso
        # regression(Lasso(), "lasso")

        # # Linear regression
        # regression(LinearRegression(), "linear")

        
        """ clustering of item index """
        kmeans = KMeans(n_clusters=env.memory_num)
        kmeans.fit(c_memorizing.reshape(-1, c_memorizing.shape[-1]))
        pred = kmeans.predict(c_memorizing.reshape(-1, c_memorizing.shape[-1]))
        rand_index = rand_score(encoding_index.reshape(-1), pred)
        adj_mutual_info = adjusted_mutual_info_score(encoding_index.reshape(-1), pred)
        
        kmeans.fit(c_recalling[ridge_mask])
        pred = kmeans.predict(c_recalling[ridge_mask])
        rand_index_rec = rand_score(recall_index[ridge_mask], pred)
        adj_mutual_info_rec = adjusted_mutual_info_score(recall_index[ridge_mask], pred)

        kmeans_stat = {
            "rand_index_enc": rand_index,
            "adj_mutual_info_enc": adj_mutual_info,
            "rand_index_rec": rand_index_rec,
            "adj_mutual_info_rec": adj_mutual_info_rec
        }
        with open(fig_path/"decoding_data"/"kmeans_stat.pkl", "wb") as f:
            pickle.dump(kmeans_stat, f)

        print()
        print("kmeans clustering")
        print("item identity encoding rand index: {}, adj mutual info: {}".format(rand_index, adj_mutual_info))
        print("item identity recall rand index: {}, adj mutual info: {}".format(rand_index_rec, adj_mutual_info_rec))





        """ decoding each previous item """
        for i in range(1, env.memory_num-1):
            # Ridge
            ridge_decoder = RidgeClassifier()
            ridge = Classifier(decoder=ridge_decoder)
            ridge_r2, ridge_acc = ridge.fit(c_memorizing[:, i:], memory_sequence[:, :-i])

            # SVM
            svm = Classifier()
            svm_r2, svm_acc = svm.fit(c_memorizing[:, i:], memory_sequence[:, :-i])

            res = {
                "ridge_r2": ridge_r2,
                "ridge_acc": ridge_acc,
                "svm_r2": svm_r2,
                "svm_acc": svm_acc
            }
            os.makedirs(fig_path/"decoding_data"/"prev_item_decode", exist_ok=True)
            with open(fig_path/"decoding_data"/"prev_item_decode"/f"item_{i}.pkl", "wb") as f:
                pickle.dump(res, f)



        """ compute statistics of weights and activity """
        weights_activity_data = {}
        
        input_weights = model.fc_input.weight.detach().cpu().numpy()
        hidden_weights = model.fc_hidden.weight.detach().cpu().numpy()
        print(input_weights.shape, hidden_weights.shape)

        i_r_weights, i_i_weights, i_n_weights = np.split(input_weights, 3, axis=0)
        h_r_weights, h_i_weights, h_n_weights = np.split(hidden_weights, 3, axis=0)

        weights_activity_data["i_r_weights_mean"] = np.mean(i_r_weights)
        weights_activity_data["i_r_weights_std"] = np.std(i_r_weights)
        weights_activity_data["i_i_weights_mean"] = np.mean(i_i_weights)
        weights_activity_data["i_i_weights_std"] = np.std(i_i_weights)
        weights_activity_data["i_n_weights_mean"] = np.mean(i_n_weights)
        weights_activity_data["i_n_weights_std"] = np.std(i_n_weights)
        weights_activity_data["h_r_weights_mean"] = np.mean(h_r_weights)
        weights_activity_data["h_r_weights_std"] = np.std(h_r_weights)
        weights_activity_data["h_i_weights_mean"] = np.mean(h_i_weights)
        weights_activity_data["h_i_weights_std"] = np.std(h_i_weights)
        weights_activity_data["h_n_weights_mean"] = np.mean(h_n_weights)
        weights_activity_data["h_n_weights_std"] = np.std(h_n_weights)


        states = []
        for i in range(context_num):
            states.append(readouts[i]['state'])
        states = np.stack(states).squeeze()
        print(states.shape)

        activity_mean_timestep = np.mean(states, axis=(0,2))
        activity_std_timestep = np.std(states, axis=(0,2))
        activity_mean = np.mean(activity_mean_timestep)
        activity_std = np.mean(activity_std_timestep)

        weights_activity_data["activity_mean"] = activity_mean
        weights_activity_data["activity_std"] = activity_std
        weights_activity_data["activity_mean_timestep"] = activity_mean_timestep
        weights_activity_data["activity_std_timestep"] = activity_std_timestep

        with open(fig_path/"weights_activity_data.pkl", "wb") as f:
            pickle.dump(weights_activity_data, f)

        print(weights_activity_data)


