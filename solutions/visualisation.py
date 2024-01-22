import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualisation(data_2d, data_3d, labels=None, save_path=None):
    
    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis', marker='o', alpha=0.5, s=10)
    plt.title('2d PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    if save_path is not None:
        plt.savefig(save_path + '_2d_pca_plot.png')
    else:
        plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], c=labels, cmap='viridis', marker='o', alpha=1.0, s=10)
    ax.set_title('3d PCA')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    if save_path is not None:
       plt.savefig(save_path + '_3d_pca_plot.png')
    else:
        plt.show()