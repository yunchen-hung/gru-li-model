import numpy as np
from scipy.stats import rankdata


class TemporalFactor:
    def __init__(self) -> None:
        self.results = None

    def fit(self, memory_contexts, actions):
        self.context_num, self.memory_num = memory_contexts.shape
        self.results = np.zeros(memory_contexts.shape)
        for i in range(self.context_num):
            recalled = np.zeros(self.memory_num, dtype=bool)
            for t in range(len(actions[i]) - 1):
                # print("t:", t)
                position1 = np.where(memory_contexts[i] == actions[i][t])
                position2 = np.where(memory_contexts[i] == actions[i][t+1])
                if position1[0].shape[0] != 0 and position2[0].shape[0] != 0 and \
                    not recalled[position1[0][0]] and not recalled[position2[0][0]]\
                        and not position1[0][0] == position2[0][0]:
                    recalled[position1[0][0]] = True
                    possible_transitions = ~np.isin(memory_contexts[i], actions[i][:t+1])
                    # print("possible_transitions:", possible_transitions)
                    actual_transition = memory_contexts[i][possible_transitions] == actions[i][t+1]
                    # print("actual_transition:", actual_transition)
                    possible_temporal_transitions = np.arange(self.memory_num)[possible_transitions] - position1[0][0]
                    # print("possible_temporal_transitions:", possible_temporal_transitions)
                    if len(possible_temporal_transitions) > 1 and np.any(actual_transition) :
                        ranks = rankdata(-np.abs(possible_temporal_transitions))
                        # print(ranks)
                        self.results[i][t+1] = (ranks[actual_transition] - 1) / (len(ranks) - 1)
                    #     print(self.results[i][t+1])
                    # print()
        return self.results
    

if __name__ == "__main__":
    memory_contexts = np.array([[33,118,192,275,261,232,231,160,72,170,174,223,249,208,3]])
    actions = np.array([[118,160,3,249,231,28,170,135,208,284,0,0,0,0,0]])
    temporal_factor = TemporalFactor()
    results = temporal_factor.fit(memory_contexts, actions)
    print("results:", results)
                    
