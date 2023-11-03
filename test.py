import torch
import matplotlib.pyplot as plt

from models.memory.similarity.lca import LCA
from models.utils import softmax


# s = torch.tensor([0.3, 0.28, 0.28, 0.1, 0.1, 0.1, 0.1, 0.1])
# s = s / torch.sum(s)
# print('original vector:\n', s)
# print('softmax vector (beta=0.1):\n', softmax(s.reshape(1, -1), beta=0.1))
# print('softmax vector (beta=0.01):\n', softmax(s.reshape(1, -1), beta=0.01))
# s = s.repeat(10, 1)
# lca = LCA(8, lateral_inhibition=0.8)
# s_out = lca(s)

# for i in range(s_out.shape[1]):
#     plt.plot(s_out[:, i])
# plt.xlabel("time")
# plt.ylabel("value")
# plt.savefig("lca.png")

# print()
# print('after LCA:\n', s_out[-1])
# print('then after normalization:\n', s_out[-1].reshape(1, -1) / torch.sum(s_out[-1]))
# print('then after softmax (beta=0.2):\n', softmax(s_out[-1].reshape(1, -1) / torch.sum(s_out[-1]), beta=0.2))

acc = [0.9416, 0.8013, 0.9893, 0.7351, 0.7825, 0.8213, 0.9256, 0.889, 0.8738, 0.9779,
       0.8229, 0.9424, 0.9303, 0.9311, 0.891, 0.8073, 0.9083, 0.8925, 0.9463, 0.9043]
forw_asym = [0.28, -0.01, 0.64, 0.02, 0.04, 0.02, 0.025, 0.023, 0.033, 0.62, 
             0.077, 0.35, 0.77, 0.42, 0.031, 0.029, 0.28, 0.017, 0.013, -0.012]

plt.figure(figsize=(4, 3))
plt.scatter(acc, forw_asym)
plt.xlabel("accuracy")
plt.ylabel("forward asymmetry")
plt.tight_layout()
plt.savefig("acc_forw_asym.png")

