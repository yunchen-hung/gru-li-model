import numpy as np
import matplotlib.pyplot as plt

from utils import savefig

class PriorListIntrusion:
    """
    calculate the prior list intrustion throughout the entire experiment
    """
    def __init__(self) -> None:
        self.results = None
    
    def fit(self, memory_contexts, actions):
        self.context_num, self.memory_num = memory_contexts.shape
        self.results = np.zeros((self.memory_num, self.memory_num))
        #self.protrusions is stored as 
        # row: output position of mistake, 
        # col: prior list study position

        # protrusions1 is for one before, protrusions2 is for two lists before
        self.protrusions1 = np.zeros((self.memory_num, self.memory_num))
        self.protrusions2 = np.zeros((self.memory_num, self.memory_num))
        for i in range(self.context_num):
            # loop through every trial
            for t in range(self.memory_num - 1):
                # loop through every item in sequence

                # if the current word is not in the study list, find in prior list
                if actions[i][t] not in memory_contexts[i]:
                    # if it is in ONE list prior
                    if actions[i][t] in memory_contexts[i-1]:
                        position = np.where(memory_contexts[i-1] == actions[i][t])
                        self.protrusions1[t][position] += 1
                    if actions[i][t] in memory_contexts[i-2]:
                        position = np.where(memory_contexts[i-2] == actions[i][t])
                        self.protrusions2[t][position] += 1
        times_sum = np.expand_dims(np.sum(self.protrusions1, axis=1), axis=1)
        times_sum[times_sum == 0] = 1
        self.protrusions1 = self.protrusions1 / times_sum

        times_sum = np.expand_dims(np.sum(self.protrusions2, axis=1), axis=1)
        times_sum[times_sum == 0] = 1
        self.protrusions2 = self.protrusions2 / times_sum

    def visualize_priorlist(self, save_path, how_many_prior= 1, 
                            save_name="serial_position_intrusion", format="png"):
        plt.figure(figsize=(4, 3.3), dpi=180)
        if how_many_prior == 1:
            protrusion_list = self.protrusions1
        if how_many_prior == 2:
            protrusion_list = self.protrusions2
        
        cmap = plt.get_cmap("tab10")
        #plot chance
        plt.hlines(1/self.memory_num, 0, self.memory_num, linestyles='dashed', colors = 'gray')
        for i in range(self.memory_num):
            color = cmap(i % 10)
            x_bar = np.arange(self.memory_num)[i] + np.arange(self.memory_num) * 0.15
            plt.plot(
                x_bar,
                protrusion_list[i,:],
                color=color,        
                linestyle='-',
            )
        plt.xlabel("Output position in Current List")
        plt.ylabel("Proportion of Errors")
        # plt.title("recall probability at each timestep")
        plt.tight_layout()
        savefig(save_path, save_name, format=format)