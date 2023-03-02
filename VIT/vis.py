"""
file containing all visualization helper functions to validate our models
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns


def visualize_attention_masks(attn_masks):
    # Get the number of attention heads and the number of patches
    num_heads, num_patches, _ = attn_masks.shape

    # Create a grid of subplots with the appropriate dimensions
    fig, axs = plt.subplots(nrows=num_heads, ncols=num_heads, figsize=(15,15))

    # Loop over all the attention heads
    for i in range(num_heads):
        for j in range(num_heads):
            # Plot the attention mask for the current pair of attention heads
            sns.heatmap(attn_masks[i, :, :, j], cmap="YlGnBu", vmin=0, vmax=1,
                        xticklabels=False, yticklabels=False, ax=axs[i,j])
            # Add labels to the subplots
            if i == 0:
                axs[i,j].set_title(f"Head {j+1}")

            if j == 0:
                axs[i,j].set_ylabel(f"Head {i+1}")
    # Add a title to the overall plot
    fig.suptitle("Attention masks for all pairs of attention heads")
    # Show the plot
    plt.show()
