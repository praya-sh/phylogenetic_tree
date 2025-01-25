from Bio import SeqIO, AlignIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class UPGMAClusterer:
    def __init__(self, distance_matrix):
        self.original_matrix = np.array(distance_matrix, dtype=float)
        self.matrix = self.original_matrix.copy()
        self.num_sequences = len(self.matrix)
        
        self.clusters = [f"Seq_{i+1}" for i in range(self.num_sequences)]
        self.cluster_sizes = np.ones(self.num_sequences, dtype=int)
        self.cluster_mapping = {}
    
    def find_minimum_distance(self):
        mask = np.triu(np.ones_like(self.matrix, dtype=bool), k=1)
        masked_matrix = np.ma.masked_array(self.matrix, mask=~mask)
        min_index = np.unravel_index(masked_matrix.argmin(), masked_matrix.shape)
        return min_index
    
    def merge_clusters(self, index1, index2):
        new_size = self.cluster_sizes[index1] + self.cluster_sizes[index2]
        new_cluster_name = f"Cluster_{len(self.clusters) + 1}"
        
        self.cluster_mapping[new_cluster_name] = {
            'left': self.clusters[index1],
            'right': self.clusters[index2],
            'left_distance': self.matrix[index1, index2] / 2,
            'right_distance': self.matrix[index1, index2] / 2
        }
        
        new_distances = np.zeros(len(self.matrix) - 1)
        for i in range(len(self.matrix)):
            if i != index1 and i != index2:
                new_dist = (
                    self.matrix[index1, i] * self.cluster_sizes[index1] + 
                    self.matrix[index2, i] * self.cluster_sizes[index2]
                ) / new_size
                new_distances[i if i < min(index1, index2) else i-1] = new_dist
        
        self.matrix = np.delete(self.matrix, [index1, index2], axis=0)
        self.matrix = np.delete(self.matrix, [index1, index2], axis=1)
        
        self.matrix = np.pad(self.matrix, ((0, 1), (0, 1)), mode='constant')
        self.matrix[-1, :-1] = new_distances[:-1]
        self.matrix[:-1, -1] = new_distances[:-1]
        
        self.clusters.append(new_cluster_name)
        self.cluster_sizes = np.delete(self.cluster_sizes, [index1, index2])
        self.cluster_sizes = np.append(self.cluster_sizes, new_size)
        
        del self.clusters[max(index1, index2)]
        del self.clusters[min(index1, index2)]
    
    def cluster(self):
        while len(self.clusters) > 1:
            index1, index2 = self.find_minimum_distance()
            self.merge_clusters(index1, index2)
        
        return self.cluster_mapping
    
    def get_newick_tree(self):
        def build_newick(cluster_name):
            if cluster_name in [f"Seq_{i+1}" for i in range(self.num_sequences)]:
                return cluster_name
            
            node = self.cluster_mapping[cluster_name]
            left = build_newick(node['left'])
            right = build_newick(node['right'])
            
            return f"({left}:{node['left_distance']},{right}:{node['right_distance']})"
        
        final_cluster = self.clusters[0]
        return build_newick(final_cluster) + ";"

class PhylogeneticWorkflow:
    def __init__(self):
        self.sequences = []
        self.alignment = None
        self.distance_matrix = None
        self.tree = None
        
    def generate_sample_data(self, num_species=5, seq_length=100, mutation_rate=0.02):
        ancestor = ''.join(random.choice('ATGC') for _ in range(seq_length))
        
        for i in range(num_species):
            seq = list(ancestor)
            mutations = int(seq_length * mutation_rate * (i + 1))
            mutation_sites = random.sample(range(seq_length), mutations)
            
            for site in mutation_sites:
                current = seq[site]
                options = [base for base in 'ATGC' if base != current]
                seq[site] = random.choice(options)
            
            record = SeqRecord(
                Seq(''.join(seq)),
                id=f"Species_{i+1}",
                description=f"Simulated sequence with {mutations} mutations"
            )
            self.sequences.append(record)
        
        SeqIO.write(self.sequences, "sample_sequences.fasta", "fasta")
        print(f"Generated {num_species} sequences and saved to sample_sequences.fasta")
    
    def create_alignment(self):
        self.alignment = MultipleSeqAlignment(self.sequences)
        AlignIO.write(self.alignment, "alignment.fasta", "fasta")
        print("Created alignment and saved to alignment.fasta")
    
    def calculate_distances(self):
        # Compute Hamming distance matrix
        num_seq = len(self.sequences)
        self.distance_matrix = np.zeros((num_seq, num_seq))
        
        for i in range(num_seq):
            for j in range(i+1, num_seq):
                # Compute Hamming distance between sequences
                diff = sum(a != b for a, b in zip(str(self.sequences[i].seq), str(self.sequences[j].seq)))
                self.distance_matrix[i, j] = diff / len(self.sequences[i].seq)
                self.distance_matrix[j, i] = self.distance_matrix[i, j]
        
        print("Calculated distance matrix using Hamming distance")
        return self.distance_matrix
    
    def construct_tree(self):
        # Use UPGMA to construct tree
        if self.distance_matrix is None:
            raise ValueError("Distance matrix not calculated. Run calculate_distances() first.")
        
        clusterer = UPGMAClusterer(self.distance_matrix)
        newick_tree = clusterer.get_newick_tree()
        
        # Write Newick tree to file
        with open("tree.nwk", "w") as f:
            f.write(newick_tree)
        
        print("Constructed UPGMA tree and saved to tree.nwk")
        
        # Optional: Parse Newick tree for further processing if needed
        from Bio import Phylo
        import io
        self.tree = Phylo.read(io.StringIO(newick_tree), "newick")
    
    def visualize_results(self):
        if self.tree is None:
            raise ValueError("Tree not constructed yet. Run construct_tree() first.")
        
        # Create figure for phylogenetic tree
        fig_tree = plt.figure(figsize=(10, 8))
        ax_tree = fig_tree.add_subplot(111)
        
        # Plot the phylogenetic tree
        Phylo.draw(self.tree, axes=ax_tree, show_confidence=False, branch_labels={n: f"{n.branch_length:.2f}" for n in self.tree.get_terminals()})
        ax_tree.set_title("Phylogenetic Tree (UPGMA)")
        ax_tree.set_xlabel("Branch Length")
        ax_tree.set_ylabel("Taxa")
        
        plt.tight_layout()
        plt.savefig("phylogenetic_tree.png")
        plt.close(fig_tree)
        
        # Distance matrix heatmap (unchanged)
        fig_heatmap = plt.figure(figsize=(10, 8))
        ax_heatmap = fig_heatmap.add_subplot(111)
        im = ax_heatmap.imshow(self.distance_matrix, cmap='viridis')
        plt.colorbar(im)
        
        species_names = [seq.id for seq in self.sequences]
        ax_heatmap.set_xticks(range(len(species_names)))
        ax_heatmap.set_yticks(range(len(species_names)))
        ax_heatmap.set_xticklabels(species_names, rotation=45)
        ax_heatmap.set_yticklabels(species_names)
        ax_heatmap.set_title("Distance Matrix Heatmap")
        
        plt.tight_layout()
        plt.savefig("distance_matrix_heatmap.png")
        plt.close(fig_heatmap)
        
        print("Saved visualizations to:")
        print("- phylogenetic_tree.png")
        print("- distance_matrix_heatmap.png")

def run_complete_analysis(num_species=5, seq_length=100):
    try:
        workflow = PhylogeneticWorkflow()
        
        print("=== Starting Phylogenetic Analysis ===")
        
        print("\n1. Generating sample sequences...")
        workflow.generate_sample_data(num_species=num_species, seq_length=seq_length)
        
        print("\n2. Creating sequence alignment...")
        workflow.create_alignment()
        
        print("\n3. Calculating distance matrix...")
        workflow.calculate_distances()
        
        print("\n4. Constructing phylogenetic tree (UPGMA)...")
        workflow.construct_tree()
        
        print("\n5. Creating visualizations...")
        workflow.visualize_results()
        
        print("\n=== Analysis Complete ===")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Please check the error message above and ensure all dependencies are installed correctly.")

if __name__ == "__main__":
    run_complete_analysis(num_species=16, seq_length=200)