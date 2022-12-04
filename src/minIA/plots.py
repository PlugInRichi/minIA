from matplotlib import pyplot as plt


def plot_principal_components_2D(principal_components, pc_centroids, path_plot):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('PCA descriptors', fontsize=20)
    ax.scatter(principal_components[:, 0], principal_components[:, 1], s=1)
    ax.scatter(pc_centroids[:, 0], pc_centroids[:, 1], s=10, c='r')
    ax.grid()
    fig.savefig(path_plot, dpi=256, bbox_inches='tight')
