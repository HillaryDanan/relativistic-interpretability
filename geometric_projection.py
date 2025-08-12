"""
Geometric Projection Operations for Relativistic Interpretability

This module implements the core projection operations that transform
attention patterns from square geometry to alternative geometric bases.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from scipy.spatial import distance_matrix
from scipy.linalg import eigh
import warnings

class GeometricProjection:
    """
    Project square attention patterns onto alternative geometric bases.
    
    Key insight: Attention patterns are weighted graphs that can be
    transformed into different geometric spaces through adjacency matrix operations.
    """
    
    def __init__(self, seq_len: int, device: str = 'cpu'):
        """
        Initialize geometric projection operators.
        
        Args:
            seq_len: Sequence length for attention patterns
            device: Computing device ('cpu' or 'cuda')
        """
        self.seq_len = seq_len
        self.device = device
        self.adjacency_matrices = self._initialize_geometries()
        self.laplacians = self._compute_laplacians()
        
    def _initialize_geometries(self) -> Dict[str, torch.Tensor]:
        """
        Create adjacency matrices for each geometry.
        
        Returns:
            Dictionary mapping geometry names to adjacency matrices
        """
        adjacencies = {}
        
        # Square (4-connectivity grid)
        square = torch.zeros(self.seq_len, self.seq_len)
        for i in range(self.seq_len):
            for j in range(self.seq_len):
                if abs(i - j) == 1:  # Adjacent positions
                    square[i, j] = 1.0
                elif abs(i - j) == int(np.sqrt(self.seq_len)):  # Grid neighbors
                    square[i, j] = 0.5
        adjacencies['square'] = square.to(self.device)
        
        # Triangular (6-connectivity)
        triangular = torch.zeros(self.seq_len, self.seq_len)
        for i in range(self.seq_len):
            # Create 6 connections per node where possible
            connections = [i-1, i+1, i-2, i+2, 
                          i-int(np.sqrt(self.seq_len)), 
                          i+int(np.sqrt(self.seq_len))]
            for j in connections:
                if 0 <= j < self.seq_len:
                    triangular[i, j] = 1.0 / 6.0
        adjacencies['triangular'] = triangular.to(self.device)
        
        # Hexagonal (optimal packing)
        hexagonal = torch.zeros(self.seq_len, self.seq_len)
        hex_radius = int(np.sqrt(self.seq_len) * 0.866)  # Hexagonal packing constant
        for i in range(self.seq_len):
            # Create hexagonal connectivity pattern
            for j in range(self.seq_len):
                dist = abs(i - j)
                if dist == 1:
                    hexagonal[i, j] = 1.0
                elif dist == hex_radius or dist == hex_radius + 1:
                    hexagonal[i, j] = 0.7
                elif dist == 2 * hex_radius:
                    hexagonal[i, j] = 0.3
        adjacencies['hexagonal'] = hexagonal.to(self.device)
        
        # Pentagonal (non-tiling, symmetry-breaking)
        pentagonal = torch.zeros(self.seq_len, self.seq_len)
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(self.seq_len):
            # Create irregular 5-fold connectivity
            connections = []
            for offset in [1, 2, int(golden_ratio * 3), int(golden_ratio * 5), int(golden_ratio * 8)]:
                if i + offset < self.seq_len:
                    connections.append(i + offset)
                if i - offset >= 0:
                    connections.append(i - offset)
            connections = connections[:5]  # Limit to 5 connections
            for j in connections:
                pentagonal[i, j] = 1.0 / len(connections)
        adjacencies['pentagonal'] = pentagonal.to(self.device)
        
        # Normalize all adjacency matrices
        for key in adjacencies:
            adj = adjacencies[key]
            row_sums = adj.sum(dim=1, keepdim=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            adjacencies[key] = adj / row_sums
            
        return adjacencies
    
    def _compute_laplacians(self) -> Dict[str, torch.Tensor]:
        """
        Compute graph Laplacians for spectral analysis.
        
        Returns:
            Dictionary mapping geometry names to Laplacian matrices
        """
        laplacians = {}
        for geom, adj in self.adjacency_matrices.items():
            degree = torch.diag(adj.sum(dim=1))
            laplacians[geom] = degree - adj
        return laplacians
    
    def project(self, attention: torch.Tensor, geometry: str) -> Dict[str, torch.Tensor]:
        """
        Project attention pattern onto specified geometry.
        
        Args:
            attention: Attention matrix [batch, heads, seq, seq]
            geometry: Target geometry ('square', 'hexagonal', 'triangular', 'pentagonal')
            
        Returns:
            Dictionary containing different projection methods
        """
        if geometry not in self.adjacency_matrices:
            raise ValueError(f"Unknown geometry: {geometry}")
            
        adj = self.adjacency_matrices[geometry]
        laplacian = self.laplacians[geometry]
        
        results = {}
        
        # Method 1: Direct filtering (element-wise multiplication)
        adj_expanded = adj.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
        filtered = attention * adj_expanded
        # Renormalize
        filtered = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-8)
        results['filtered'] = filtered
        
        # Method 2: Spectral projection
        # Compute eigenvectors of the Laplacian
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
        # Project attention onto eigenvector basis
        spectral = torch.zeros_like(attention)
        for b in range(attention.shape[0]):
            for h in range(attention.shape[1]):
                att_matrix = attention[b, h]
                # Project onto top-k eigenvectors
                k = min(20, self.seq_len // 2)
                proj = eigenvectors[:, :k] @ eigenvectors[:, :k].T @ att_matrix
                spectral[b, h] = proj
        results['spectral'] = spectral
        
        # Method 3: Message passing (graph convolution)
        flow = attention.clone()
        for _ in range(3):  # 3 iterations of message passing
            flow = torch.matmul(adj_expanded, flow)
            flow = F.softmax(flow, dim=-1)
        results['flow'] = flow
        
        return results
    
    def measure_affinity(self, attention: torch.Tensor, geometry: str) -> float:
        """
        Measure how well attention pattern fits specified geometry.
        
        Args:
            attention: Attention matrix [batch, heads, seq, seq]
            geometry: Target geometry
            
        Returns:
            Affinity score (0-1, higher is better fit)
        """
        if geometry not in self.adjacency_matrices:
            raise ValueError(f"Unknown geometry: {geometry}")
            
        adj = self.adjacency_matrices[geometry]
        
        # Compute multiple affinity metrics
        affinities = []
        
        # 1. Structural overlap
        attention_mean = attention.mean(dim=(0, 1))  # Average over batch and heads
        overlap = (attention_mean * adj).sum() / (attention_mean.sum() + 1e-8)
        affinities.append(overlap.item())
        
        # 2. Spectral similarity
        laplacian = self.laplacians[geometry]
        att_laplacian = torch.diag(attention_mean.sum(dim=1)) - attention_mean
        
        # Compare eigenvalue distributions
        eig_adj = torch.linalg.eigvalsh(laplacian)
        eig_att = torch.linalg.eigvalsh(att_laplacian)
        
        # Normalize eigenvalues
        eig_adj = eig_adj / (eig_adj.max() + 1e-8)
        eig_att = eig_att / (eig_att.max() + 1e-8)
        
        # Compute similarity (1 - normalized distance)
        spectral_sim = 1 - torch.norm(eig_adj - eig_att) / np.sqrt(self.seq_len)
        affinities.append(spectral_sim.item())
        
        # 3. Flow preservation
        flow_adj = adj @ attention_mean
        flow_similarity = F.cosine_similarity(
            flow_adj.flatten().unsqueeze(0),
            attention_mean.flatten().unsqueeze(0)
        )
        affinities.append(flow_similarity.item())
        
        # Return weighted average
        weights = [0.3, 0.4, 0.3]  # Weights for each metric
        final_affinity = sum(w * a for w, a in zip(weights, affinities))
        
        return max(0, min(1, final_affinity))  # Clamp to [0, 1]
    
    def compute_geometric_entropy(self, affinities: Dict[str, float]) -> float:
        """
        Compute entropy over geometric affinities.
        
        Args:
            affinities: Dictionary mapping geometry names to affinity scores
            
        Returns:
            Entropy value (higher = more uniform distribution)
        """
        values = torch.tensor(list(affinities.values()))
        values = values / values.sum()  # Normalize to probability distribution
        
        # Compute entropy
        entropy = -(values * torch.log(values + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(len(affinities), dtype=torch.float))
        
        return (entropy / max_entropy).item()  # Normalized entropy [0, 1]
    
    def identify_dominant_geometry(self, attention: torch.Tensor) -> Tuple[str, Dict[str, float]]:
        """
        Identify which geometry best describes the attention pattern.
        
        Args:
            attention: Attention matrix
            
        Returns:
            Tuple of (dominant geometry name, all affinity scores)
        """
        affinities = {}
        for geometry in self.adjacency_matrices.keys():
            affinities[geometry] = self.measure_affinity(attention, geometry)
        
        dominant = max(affinities, key=affinities.get)
        return dominant, affinities