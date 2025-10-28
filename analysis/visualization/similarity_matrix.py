import matplotlib.pyplot as plt
from utils import savefig

def SimilarityBetweenStates(similarity, fig_path, xlab, ylab, plot_name):
    plt.figure(figsize=(4.5, 3.7), dpi=180)
    plt.imshow(similarity, cmap="Blues")
    plt.colorbar(label="cosine similarity\nbetween hidden states")
    plt.xlabel(f"time in {xlab} phase")
    plt.ylabel(f"time in {ylab} phase")
    # set the color bar to be between 0 and 1
    plt.clim(0, 1)  # set color limits to [0, 1]
    # plt.title("encoding-recalling state similarity")
    plt.tight_layout()
    savefig(fig_path/"state_similarity", plot_name)