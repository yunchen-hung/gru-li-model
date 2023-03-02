import numpy as np
import torch
import matplotlib.pyplot as plt

from utils import savefig
from analysis.decomposition import PCA


class FixedPointAnalysis:
    def __init__(self):
        self.fixed_points = None
        self.pca = PCA(n_components=2)

    def fit(self, model, dataset, sample_num=10, time_step=10000):
        model.set_encoding(False)
        model.set_retrieval(False)

        for param in model.parameters():
            param.requires_grad = False
        self.fixed_points = []
        for p in range(sample_num):
            model.hidden_state = torch.tensor(np.random.rand(1, model.hidden_dim)*10,
                        requires_grad=True, dtype=torch.float32)
            hidden = torch.tensor(np.random.rand(1, model.hidden_dim)*10,
                        requires_grad=True, dtype=torch.float32)
            # print(hidden.is_leaf, hidden.shape)

            optimizer = torch.optim.Adam([hidden], lr=0.001)
            criterion = torch.nn.MSELoss()

            input = torch.zeros(1, model.input_dim)

            running_loss = 0
            print("Finding fixed points...")
            for i in range(time_step):
                optimizer.zero_grad()   # zero the gradient buffers
                
                # Take the one-step recurrent function from the trained network
                _, _, new_h = model.forward(input, hidden)
                loss = criterion(new_h, hidden)
                loss.backward()
                optimizer.step()    # Does the update

                running_loss += loss.item()
                if i % 1000 == 999:
                    running_loss /= 1000
                    print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
                    running_loss = 0

            self.fixed_points.append(hidden.detach().numpy())
        self.fixed_points = np.stack(self.fixed_points, axis=0)

        self.activity_pc = self.pca.fit_transform(dataset)
        self.fixed_points_pc = self.pca.pca.transform(self.fixed_points.squeeze())

    def visualize(self, save_path=None, pdf=False):
        self.pca.visualize_state_space(constrain_lim=False)
        print(self.fixed_points_pc.shape)
        plt.scatter(self.fixed_points_pc[:, 0], self.fixed_points_pc[:, 1], c='k', marker='x', s=10)
        if save_path is not None:
            savefig(save_path, "fixed_points", pdf=pdf)