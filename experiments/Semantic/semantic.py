
import os
import numpy as np
import sklearn.metrics.pairwise as skp
import matplotlib.pyplot as plt

from analysis.behavior import SemanticContiguity
from utils import savefig



def run(data_all, model_all, env, paths, exp_name, **kwargs):
    plt.rcParams['font.size'] = 18

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

        memory_num = env.unwrapped.sequence_len

        model = model_all[run_name]
        if hasattr(model, "step_for_each_timestep"):
            step_for_each_timestep = model.step_for_each_timestep
            timestep_each_phase = step_for_each_timestep * memory_num
        else:
            step_for_each_timestep = 1
            timestep_each_phase = memory_num
        # timestep_each_phase = env.memory_num

        # get recorded data and outputs of the model
        readouts = data['readouts']
        actions = data['actions']
        rewards = data['rewards']

        all_context_num = len(actions)
        context_num = min(all_context_num, 20)

        def convert_actions(actions):
            feature_num = env.unwrapped.num_features
            feature_dim = env.unwrapped.feature_dim
            converted_actions = np.zeros((actions.shape[0], actions.shape[1], feature_num+2))
            action_space_shape = feature_dim ** feature_num + 1
            for i in range(actions.shape[0]):
                for j in range(actions.shape[1]):
                    if actions[i][j][0] == action_space_shape - 1:
                        # no action
                        converted_actions[i][j][-1] = 1
                    else:
                        converted_actions[i][j][:feature_num] = np.array([(actions[i][j][0] - 1) // (feature_dim ** k) % feature_dim for k in range(feature_num)])
            return converted_actions
        
        print("actions raw shape: ", np.array(actions).shape)
        print("exp name: ", exp_name)
        actions = np.array(actions).squeeze(-1)
        actions = convert_actions(actions)

        rewards = np.array(rewards)
        rewards = rewards.squeeze()
        rewards = rewards.reshape(-1, rewards.shape[-1])        # (trials, timesteps per trial)
        print("action shape: ", actions.shape)

        print(actions.shape, rewards.shape)



        """ semantic contiguity """
        memory_contexts = []
        for i in range(all_context_num):
            memory_contexts.append(data['trial_data'][i]["memory_sequence_int"])
        memory_contexts = np.array(memory_contexts)     # ground truth of memory for each trial
        memory_contexts = memory_contexts.reshape(-1, memory_contexts.shape[-1])    # reshape to (trials, sequence_len)
        print("memory_contexts shape: ", memory_contexts.shape)

        memory_contexts_features = []
        for i in range(all_context_num):
            memory_contexts_features.append(data['trial_data'][i]["memory_sequence"])
        memory_contexts_features = np.array(memory_contexts_features)
        print("memory_contexts_features shape: ", memory_contexts_features.shape)
        # Create permuted version of memory contexts by independently shuffling each trial's sequence
        memory_contexts_features_permuted = memory_contexts_features.copy()
        # for i in range(memory_contexts_features.shape[0]):
        #     perm = np.random.permutation(memory_contexts_features.shape[1])
        #     memory_contexts_features_permuted[i] = memory_contexts_features[i][perm]

        print(actions[:, :, :env.unwrapped.num_features].shape, memory_contexts_features_permuted.shape)
        print(actions[0, timestep_each_phase:, :env.unwrapped.num_features])
        print(memory_contexts_features_permuted[0])

        semantic_contiguity = SemanticContiguity()
        results = semantic_contiguity.fit(actions[:, timestep_each_phase:, :env.unwrapped.num_features], env.unwrapped.feature_dim)
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_normalized", use_normalized=True, title="semantic contiguity", format="png")
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity", use_normalized=False, title="semantic contiguity", format="png")

        results_gt = semantic_contiguity.fit(memory_contexts_features_permuted, env.unwrapped.feature_dim)
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_gt_normalized", use_normalized=True, title="semantic contiguity", format="png")
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_gt", use_normalized=False, title="semantic contiguity", format="png")

        print(results, results_gt)
        # results_gt[results_gt == 0] = 1
        semantic_contiguity.results = results / results_gt
        semantic_contiguity.visualize(fig_path, save_name="semantic_contiguity_norm_ratio", use_normalized=False, title="semantic contiguity", format="png")

        os.makedirs(fig_path/"data", exist_ok=True)
        np.save(fig_path/"data"/"semantic_contiguity_results.npy", results)
        np.save(fig_path/"data"/"semantic_contiguity_results_gt.npy", results_gt)
        np.save(fig_path/"data"/"semantic_contiguity_norm_ratio.npy", semantic_contiguity.results)


        """ do the hidden state get away from the just recalled item? """
        # get all the data needed
        rec_states = []
        retrieved_memories = []
        most_similar_memories = []
        for i in range(all_context_num):
            state = readouts[i]['state'].squeeze()
            rec_states.append(state[-timestep_each_phase:])
            retrieved_memories.append(readouts[i]['ValueMemory']['retrieved_memory'].squeeze())
            memory_similarity = readouts[i]['ValueMemory']['similarity']
            most_similar_index = np.argmax(memory_similarity, axis=-1).squeeze()
            most_similar_memories.append(state[most_similar_index])
        rec_states = np.stack(rec_states)
        retrieved_memories = np.stack(retrieved_memories)
        most_similar_memories = np.stack(most_similar_memories)
        print("rec_states shape: ", rec_states.shape)
        print("retrieved_memories shape: ", retrieved_memories.shape)

        # calculate the distance between the hidden state and the just recalled memory
        distances = np.zeros((timestep_each_phase, timestep_each_phase))
        for i in range(timestep_each_phase):
            for j in range(timestep_each_phase):
                dist = 0.0
                for k in range(all_context_num):
                    x = rec_states[k][i] / np.linalg.norm(rec_states[k][i])
                    y = retrieved_memories[k][j] / np.linalg.norm(retrieved_memories[k][j])
                    # print(x.shape, y.shape)
                    if np.sum(x * y) > 1:
                        print("strange cosine similarity: ", np.sum(x * y))
                    dist += np.sum(x * y)
                distances[i][j] = dist / all_context_num
        # distances = distances / np.sum(np.abs(distances), axis=-1, keepdims=True)
        
        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(distances, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(label="cosine similarity")
        plt.xlabel("retrieved memory")
        plt.ylabel("time in recall phase")
        plt.tight_layout()
        savefig(fig_path/"distances", "state_retrieved_memory.png")


        # calculate the distance between the hidden state and the most similar memory
        distances = np.zeros((timestep_each_phase, timestep_each_phase))
        for i in range(timestep_each_phase):
            for j in range(timestep_each_phase):
                dist = 0.0
                for k in range(all_context_num):
                    x = rec_states[k][i] / np.linalg.norm(rec_states[k][i])
                    y = most_similar_memories[k][j] / np.linalg.norm(most_similar_memories[k][j])
                    if np.sum(x * y) > 1:
                        print("strange cosine similarity: ", np.sum(x * y))
                    dist += np.sum(x * y)
                distances[i][j] = dist / all_context_num
        # distances = distances / np.sum(np.abs(distances), axis=-1, keepdims=True)
        
        plt.figure(figsize=(4.5, 3.7), dpi=180)
        plt.imshow(distances, cmap="RdBu", vmin=-1, vmax=1)
        plt.colorbar(label="cosine similarity")
        plt.xlabel("most similar memory")
        plt.ylabel("time in recall phase")
        plt.tight_layout()
        savefig(fig_path/"distances", "state_most_similar_memory.png")
