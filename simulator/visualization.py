"""Visualization utilities
"""
import networkx as nx
import torch_geometric as pyg
import matplotlib.pyplot as plt


def visualize_losses(losses, labels, save_path, params):
    # visualize the loss curve
    plt.figure()
    for loss, label in zip(losses, labels):
        plt.plot(*zip(*loss), label=label)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.semilogy()
    plt.tight_layout()
    plt.savefig(save_path, dpi=params.dpi, format=params.save_format)


def visualize_graph(graph, position, save_path, params):
    # remove directions of edges, because it is a symmetric directed graph.
    nx_graph = pyg.utils.to_networkx(graph).to_undirected()
    plt.figure(figsize=(7, 7))
    nx.draw(nx_graph, pos={i: tuple(v) for i, v in enumerate(position)}, node_size=50)
    plt.tight_layout()
    plt.savefig(save_path, dpi=params.dpi, format=params.save_format)

