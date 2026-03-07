import torch
from matplotlib import pyplot as plt


def plot_optimization_results(result):
    """Plot optimization history"""
    if not result.history:
        print("No history to plot")
        return

    history = torch.stack([torch.stack(h) for h in result.history])

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))

    # Plot MRL1
    axes[0, 0].plot(history[:, 0].cpu().numpy())
    axes[0, 0].set_title('MRL1 over iterations')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('MRL1')

    # Plot MRL2
    axes[0, 1].plot(history[:, 1].cpu().numpy())
    axes[0, 1].set_title('MRL2 over iterations')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('MRL2')

    # Plot Protein Loss
    axes[1, 0].plot(history[:, 2].cpu().numpy())
    axes[1, 0].set_title('Protein Loss over iterations')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')

    # Plot Number of Differences
    axes[1, 1].plot(history[:, 3].cpu().numpy())
    axes[1, 1].set_title('Sequence Changes over iterations')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Number of differences')

    # Plot Edit Loss
    axes[2, 0].plot(history[:, 4].cpu().numpy())
    axes[2, 0].set_title('Edit Loss over iterations')
    axes[2, 0].set_xlabel('Iteration')
    axes[2, 0].set_ylabel('Edit Loss')

    plt.tight_layout()
    plt.show()
