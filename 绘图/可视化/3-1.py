import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from the text file
name = "lastfm"  # Ensure the correct file extension is used
data = np.loadtxt(name)
data_recall = data[:,::2]
data_ndcg = data[:,1::2]
# Define model names
model_names = ["cke", "rippleNet", "kgat", "kgin", "kgcf"]

# Set up the figure
fig, ax1 = plt.subplots(figsize=(14, 8))

# Create a bar width
bar_width = 0.2
num_k = data_recall.shape[1]
x = np.arange(len(model_names))  # X locations for the models

# Calculate the total width of all bars for one model
total_width = bar_width * num_k
ks = [10,20,40]
# Plot each k value's performance for each model (Recall)
for k in range(num_k)[::-1]:  # Loop through each k value
    performance = data_recall[:, k]  # Get performance for the current k value
    offset = (k - (num_k - 1) / 2) * bar_width  # Center the group of bars
    ax1.bar(x + offset, performance, width=bar_width, label=f'Recall@{ks[k]}', alpha=0.7)

# Customize the left y-axis (Recall)
ax1.set_ylabel('Recall@K', fontsize=14)
ax1.set_title('Amazon-Book Performance', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim(bottom=0)

# Create the right y-axis for NDCG
ax2 = ax1.twinx()

# Plot NDCG values as lines
colors = plt.cm.rainbow(np.linspace(0, 1, num_k))
for k in range(num_k):
    performance = data_ndcg[:, k]
    ax2.plot(x, performance, marker='o', linestyle='-', linewidth=2, markersize=8,
             label=f'NDCG@{ks[k]}', color=colors[k])

# Customize the right y-axis (NDCG)
ax2.set_ylabel('NDCG@K', fontsize=14)
ax2.set_ylim(bottom=0)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

# Create a new axes for the legend
legend_ax = fig.add_axes([0.09, 0.75, 0.3, 0.2])  # [left, bottom, width, height]
legend_ax.axis('off')  # Turn off axis lines and labels

# Add the combined legend to the new axes
# legend = legend_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', ncol=2, fontsize=10)
legend = legend_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper center',ncol=3, fontsize=12)
legend.get_frame().set_alpha(0.8)  # Set legend background transparency

# Add a common x-label

# Adjust layout and display
plt.tight_layout()
plt.savefig(f"./data/{name}.png", dpi=600)
plt.show()