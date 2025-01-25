from Bio import SeqIO, AlignIO, Phylo
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt



class PhylogeneticWorkflow:
    def __init__(self):
        self.sequences = []
        self.alignment = None
        self.calculator = None
        self.distance_matrix = None
        self.tree = None
        
    def generate_sample_data(self, num_species=5, seq_length=100, mutation_rate=0.02):
        """Generate sample sequence data with controlled mutations"""
        # Create ancestral sequence
        ancestor = ''.join(random.choice('ATGC') for _ in range(seq_length))
        
        # Generate evolved sequences
        for i in range(num_species):
            # Introduce mutations
            seq = list(ancestor)
            mutations = int(seq_length * mutation_rate * (i + 1))
            mutation_sites = random.sample(range(seq_length), mutations)
            
            for site in mutation_sites:
                current = seq[site]
                options = [base for base in 'ATGC' if base != current]
                seq[site] = random.choice(options)
            
            # Create sequence record
            record = SeqRecord(
                Seq(''.join(seq)),
                id=f"Species_{i+1}",
                description=f"Simulated sequence with {mutations} mutations"
            )
            self.sequences.append(record)
        
        SeqIO.write(self.sequences, "sample_sequences.fasta", "fasta")
        print(f"Generated {num_species} sequences and saved to sample_sequences.fasta")
    
    def create_alignment(self):
        """Create multiple sequence alignment"""
        aligned_sequences = []
        for record in self.sequences:
            aligned_sequences.append(record)
        
        self.alignment = MultipleSeqAlignment(aligned_sequences)
        AlignIO.write(self.alignment, "alignment.fasta", "fasta")
        print("Created alignment and saved to alignment.fasta")
    
    def calculate_distances(self, model="identity"):
        """Calculate distance matrix"""
        self.calculator = DistanceCalculator(model)
        self.distance_matrix = self.calculator.get_distance(self.alignment)
        print(f"Calculated distance matrix using {model} model")
    
    def construct_tree(self):
        """Construct phylogenetic tree"""
        if self.calculator is None:
            raise ValueError("Distance calculator not initialized. Run calculate_distances() first.")
            
        constructor = DistanceTreeConstructor(self.calculator)
        self.tree = constructor.build_tree(self.alignment)
        
        Phylo.write(self.tree, "tree.nwk", "newick")
        print("Constructed tree and saved to tree.nwk")

    def visualize_results(self):
        """Visualize the phylogenetic tree and create basic statistics"""
        try:
            if self.tree is None:
                raise ValueError("Tree not constructed yet. Run construct_tree() first.")
            
            # Create separate figures for tree and heatmap
            # Tree visualization
            fig_tree = plt.figure(figsize=(10, 8))
            ax_tree = fig_tree.add_subplot(111)
            Phylo.draw(self.tree, axes=ax_tree, show_confidence=False)
            ax_tree.set_title("Phylogenetic Tree")
            plt.savefig("phylogenetic_tree.png")
            plt.close(fig_tree)
            
            # Distance matrix heatmap
            fig_heatmap = plt.figure(figsize=(10, 8))
            ax_heatmap = fig_heatmap.add_subplot(111)
            matrix_data = np.array([[float(val) for val in row] for row in self.distance_matrix])
            im = ax_heatmap.imshow(matrix_data, cmap='viridis')
            plt.colorbar(im)
            
            # Add labels
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
            
        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            print("Attempting to save tree in text format...")
            try:
                Phylo.draw_ascii(self.tree)
            except:
                print("Unable to create ASCII tree visualization")

def load_and_align_sequences(self, file_path):
    """Load sequences, align, and truncate to consistent length"""
    # Load sequences
    raw_sequences = list(SeqIO.parse(file_path, "fasta"))
    
    # Check sequence lengths
    seq_lengths = [len(seq) for seq in raw_sequences]
    
    # Truncate to minimum length
    min_length = min(seq_lengths)
    
    # Truncate sequences
    self.sequences = [
        SeqRecord(
            Seq(str(seq.seq)[:min_length]),  # Truncate sequence
            id=seq.id,
            description=f"Truncated to {min_length} bases"
        ) for seq in raw_sequences
    ]
    
    print(f"Loaded {len(self.sequences)} sequences")
    print(f"Truncated to consistent length of {min_length} bases")

# Replace the previous load_sequences method
PhylogeneticWorkflow.load_sequences = load_and_align_sequences

# Update run_complete_analysis to use the new method
def run_complete_analysis(fasta_file='sequences.fasta'):
    """Run complete phylogenetic analysis workflow with real dataset"""
    try:
        workflow = PhylogeneticWorkflow()
        
        print("=== Starting Phylogenetic Analysis ===")
        
        print("\n1. Loading and aligning sequences...")
        workflow.load_sequences(fasta_file)  # Note the method name is still load_sequences

        # print("\n1. Generating sample sequences...")
        # workflow.generate_sample_data(num_species=num_species, seq_length=seq_length)
        
        
        print("\n2. Creating sequence alignment...")
        workflow.create_alignment()
        
        print("\n3. Calculating distance matrix...")
        workflow.calculate_distances(model="identity")
        
        print("\n4. Constructing phylogenetic tree...")
        workflow.construct_tree()
        
        print("\n5. Creating visualizations...")
        workflow.visualize_results()
        
        print("\n=== Analysis Complete ===")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        print("Please check the FASTA file format and sequence alignment.")

if __name__ == "__main__":
    run_complete_analysis('sequences.fasta')
    #run_complete_analysis(num_species=16, seq_length=200)