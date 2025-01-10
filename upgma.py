import numpy as np
from collections import defaultdict

class Node:
    def __init__(self, name, height=0.0):
        self.name = name
        self.height = height
        self.children = []

def upgma(distance_matrix, labels):
    """
    Implements the UPGMA algorithm for phylogenetic tree construction.
    
    Parameters:
    distance_matrix: numpy array of pairwise distances
    labels: list of taxa names
    
    Returns:
    root: Root node of the constructed phylogenetic tree
    """
    n = len(labels)
    # Initialize clusters with leaf nodes
    clusters = [Node(label) for label in labels]
    
    # Keep track of current distances between clusters
    distances = distance_matrix.copy()
    
    while len(clusters) > 1:
        # Set diagonal to infinity to avoid selecting it as minimum
        np.fill_diagonal(distances, np.inf)
        
        # Find minimum distance
        min_dist = np.min(distances)
        min_i, min_j = np.where(distances == min_dist)[0][0], np.where(distances == min_dist)[1][0]
        
        # Ensure min_i is smaller than min_j
        if min_i > min_j:
            min_i, min_j = min_j, min_i
            
        # Calculate height (average distance / 2)
        height = distances[min_i, min_j] / 2.0
        
        # Create new node
        new_node = Node(f"Node_{min_i}_{min_j}", height)
        new_node.children = [clusters[min_i], clusters[min_j]]
        
        # Create new distance matrix with size reduced by 1
        new_size = len(clusters) - 1
        new_distances = np.zeros((new_size, new_size))
        
        # Map to keep track of new matrix indices
        old_to_new = {}
        new_idx = 0
        for old_idx in range(len(clusters)):
            if old_idx != min_i and old_idx != min_j:
                old_to_new[old_idx] = new_idx
                new_idx += 1
        
        # Fill in the new distance matrix
        for i in range(len(clusters)):
            if i == min_i or i == min_j:
                continue
            for j in range(i + 1, len(clusters)):
                if j == min_i or j == min_j:
                    continue
                new_i, new_j = old_to_new[i], old_to_new[j]
                new_distances[new_i, new_j] = distances[i, j]
                new_distances[new_j, new_i] = distances[i, j]
        
        # Calculate and fill in distances to the new cluster
        for i in range(len(clusters)):
            if i != min_i and i != min_j:
                new_i = old_to_new[i]
                dist_to_new = (distances[i, min_i] + distances[i, min_j]) / 2.0
                new_distances[new_i, new_size - 1] = dist_to_new
                new_distances[new_size - 1, new_i] = dist_to_new
        
        # Update clusters and distances
        clusters = [c for idx, c in enumerate(clusters) if idx not in (min_i, min_j)]
        clusters.append(new_node)
        distances = new_distances
        
    return clusters[0]

def print_tree(node, level=0):
    """
    Print the phylogenetic tree in a hierarchical format.
    """
    indent = "  " * level
    print(f"{indent}{node.name} (height: {node.height:.2f})")
    for child in node.children:
        print_tree(child, level + 1)

def example_usage():
    # Example distance matrix and labels
    labels = ['A', 'B', 'C', 'D']
    distance_matrix = np.array([
        [0.0, 4.0, 7.0, 6.0],
        [4.0, 0.0, 7.0, 6.0],
        [7.0, 7.0, 0.0, 5.0],
        [6.0, 6.0, 5.0, 0.0]
    ])

    # Run UPGMA
    root = upgma(distance_matrix, labels)
    
    # Print the resulting tree
    print("Phylogenetic Tree:")
    print_tree(root)

if __name__ == "__main__":
    example_usage()