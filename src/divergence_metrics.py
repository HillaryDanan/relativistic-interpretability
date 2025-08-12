"""
Geometric Divergence Metrics for Relativistic Interpretability

Measures disagreement between different geometric interpretations.
These metrics quantify when and how different geometric lenses
provide conflicting views of the same neural computation.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import spearmanr, entropy
from scipy.spatial.distance import jensenshannon
import warnings

class GeometricDivergenceMetrics:
    """
    Concrete metrics for measuring geometric disagreement.
    
    Key insight: High divergence indicates that the neural network's
    computation has multiple valid interpretations - like a quantum
    superposition of geometric states.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize divergence metric calculator.
        
        Args:
            device: Computing device ('cpu' or 'cuda')
        """
        self.device = device
        
    def compute_divergence(self, geometric_outputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute comprehensive divergence metrics between geometric interpretations.
        
        Args:
            geometric_outputs: Dictionary mapping geometry names to output tensors
                              Each tensor shape: [batch, heads, seq, seq]
            
        Returns:
            Dictionary of divergence metrics
        """
        metrics = {}
        
        # Get list of geometries and their outputs
        geometries = list(geometric_outputs.keys())
        if len(geometries) < 2:
            warnings.warn("Need at least 2 geometries to compute divergence")
            return {'error': 'insufficient_geometries'}
        
        # 1. KL Divergence between attention distributions
        kl_divergences = []
        for i, geom1 in enumerate(geometries):
            for geom2 in geometries[i+1:]:
                kl_div = self._compute_kl_divergence(
                    geometric_outputs[geom1],
                    geometric_outputs[geom2]
                )
                kl_divergences.append(kl_div)
        
        metrics['kl_divergence_mean'] = np.mean(kl_divergences) if kl_divergences else 0.0
        metrics['kl_divergence_max'] = np.max(kl_divergences) if kl_divergences else 0.0
        
        # 2. Rank Correlation of top-k features
        rank_correlations = []
        for i, geom1 in enumerate(geometries):
            for geom2 in geometries[i+1:]:
                rank_corr = self._compute_rank_correlation(
                    geometric_outputs[geom1],
                    geometric_outputs[geom2],
                    top_k=10
                )
                rank_correlations.append(rank_corr)
        
        metrics['rank_correlation_mean'] = np.mean(rank_correlations) if rank_correlations else 0.0
        metrics['rank_correlation_min'] = np.min(rank_correlations) if rank_correlations else 0.0
        
        # 3. Geometric Routing Entropy
        routing_entropy = self._compute_routing_entropy(geometric_outputs)
        metrics['routing_entropy'] = routing_entropy
        
        # 4. Phase Coherence (simulated - would connect to Ouroboros in full implementation)
        phase_coherence = self._compute_phase_coherence(geometric_outputs)
        metrics['phase_coherence'] = phase_coherence
        
        # 5. Jensen-Shannon Divergence (symmetric version of KL)
        js_divergences = []
        for i, geom1 in enumerate(geometries):
            for geom2 in geometries[i+1:]:
                js_div = self._compute_js_divergence(
                    geometric_outputs[geom1],
                    geometric_outputs[geom2]
                )
                js_divergences.append(js_div)
        
        metrics['js_divergence_mean'] = np.mean(js_divergences) if js_divergences else 0.0
        
        # 6. Attention Focus Divergence (how concentrated vs dispersed)
        focus_divergence = self._compute_focus_divergence(geometric_outputs)
        metrics['focus_divergence'] = focus_divergence
        
        return metrics
    
    def _compute_kl_divergence(self, output1: torch.Tensor, output2: torch.Tensor) -> float:
        """
        Compute KL divergence between two attention distributions.
        
        Args:
            output1, output2: Attention tensors [batch, heads, seq, seq]
            
        Returns:
            Average KL divergence
        """
        # Flatten to probability distributions
        p = output1.flatten()
        q = output2.flatten()
        
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p = p + eps
        q = q + eps
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Compute KL divergence
        kl = (p * torch.log(p / q)).sum()
        
        return kl.item()
    
    def _compute_js_divergence(self, output1: torch.Tensor, output2: torch.Tensor) -> float:
        """
        Compute Jensen-Shannon divergence (symmetric KL).
        
        Args:
            output1, output2: Attention tensors
            
        Returns:
            JS divergence value
        """
        # Convert to numpy and flatten
        p = output1.detach().cpu().numpy().flatten()
        q = output2.detach().cpu().numpy().flatten()
        
        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        # Compute JS divergence
        js_div = jensenshannon(p, q) ** 2  # Square to get JS divergence
        
        return float(js_div)
    
    def _compute_rank_correlation(self, output1: torch.Tensor, output2: torch.Tensor, 
                                  top_k: int = 10) -> float:
        """
        Compute rank correlation of top-k attended positions.
        
        Args:
            output1, output2: Attention tensors
            top_k: Number of top positions to consider
            
        Returns:
            Spearman rank correlation coefficient
        """
        # Get attention scores for each position
        scores1 = output1.mean(dim=(0, 1)).flatten()  # Average over batch and heads
        scores2 = output2.mean(dim=(0, 1)).flatten()
        
        # Get top-k indices
        _, top_indices1 = torch.topk(scores1, min(top_k, len(scores1)))
        _, top_indices2 = torch.topk(scores2, min(top_k, len(scores2)))
        
        # Create rank arrays
        ranks1 = torch.zeros(len(scores1))
        ranks2 = torch.zeros(len(scores2))
        
        for rank, idx in enumerate(top_indices1):
            ranks1[idx] = rank
        for rank, idx in enumerate(top_indices2):
            ranks2[idx] = rank
        
        # Compute Spearman correlation
        ranks1_np = ranks1.cpu().numpy()
        ranks2_np = ranks2.cpu().numpy()
        
        correlation, _ = spearmanr(ranks1_np, ranks2_np)
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def _compute_routing_entropy(self, geometric_outputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute entropy of geometric routing decisions.
        
        High entropy = no clear geometric preference
        Low entropy = strong geometric preference
        
        Args:
            geometric_outputs: Dictionary of geometric projections
            
        Returns:
            Routing entropy value
        """
        # Compute "routing probabilities" based on attention magnitude
        routing_scores = {}
        
        for geom, output in geometric_outputs.items():
            # Use L2 norm of attention as routing score
            score = torch.norm(output, p=2).item()
            routing_scores[geom] = score
        
        # Convert to probability distribution
        total = sum(routing_scores.values()) + 1e-10
        routing_probs = np.array([score / total for score in routing_scores.values()])
        
        # Compute entropy
        routing_entropy = entropy(routing_probs, base=2)
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(geometric_outputs))
        normalized_entropy = routing_entropy / max_entropy if max_entropy > 0 else 0
        
        return float(normalized_entropy)
    
    def _compute_phase_coherence(self, geometric_outputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute phase coherence with Ouroboros cycles.
        
        This is a placeholder that would connect to the Ouroboros framework.
        In the full implementation, this would measure how well the geometric
        patterns align with the cyclic phases.
        
        Args:
            geometric_outputs: Dictionary of geometric projections
            
        Returns:
            Phase coherence score
        """
        # Simulated phase coherence based on geometric consistency
        # In reality, this would interface with Ouroboros phase detection
        
        coherence_scores = []
        
        # Expected phase-geometry mapping
        phase_geometry_map = {
            'square': 0.452,      # Analysis phase
            'triangular': 0.271,  # Synthesis phase
            'hexagonal': 0.180,   # Reflection phase
            'pentagonal': 0.097   # Transformation phase
        }
        
        for geom, output in geometric_outputs.items():
            if geom in phase_geometry_map:
                # Measure how well this geometry matches expected phase distribution
                expected = phase_geometry_map[geom]
                actual = torch.norm(output, p=1).item() / (output.numel() + 1e-10)
                coherence = 1 - abs(expected - actual)
                coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.5
    
    def _compute_focus_divergence(self, geometric_outputs: Dict[str, torch.Tensor]) -> float:
        """
        Compute divergence in attention focus patterns.
        
        Measures whether different geometries focus on same vs different regions.
        
        Args:
            geometric_outputs: Dictionary of geometric projections
            
        Returns:
            Focus divergence score
        """
        focus_patterns = {}
        
        for geom, output in geometric_outputs.items():
            # Compute attention focus (center of mass)
            attention_sum = output.mean(dim=(0, 1))  # Average over batch and heads
            
            # Get focus statistics
            max_val = attention_sum.max().item()
            mean_val = attention_sum.mean().item()
            std_val = attention_sum.std().item()
            
            # Focus score: high when concentrated, low when dispersed
            focus_score = (max_val - mean_val) / (std_val + 1e-10)
            focus_patterns[geom] = focus_score
        
        # Compute variance in focus patterns
        focus_values = list(focus_patterns.values())
        focus_divergence = np.std(focus_values) / (np.mean(focus_values) + 1e-10)
        
        return float(focus_divergence)
    
    def compute_pairwise_divergences(self, geometric_outputs: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str], float]:
        """
        Compute all pairwise divergences between geometries.
        
        Args:
            geometric_outputs: Dictionary of geometric projections
            
        Returns:
            Dictionary mapping geometry pairs to divergence scores
        """
        pairwise = {}
        geometries = list(geometric_outputs.keys())
        
        for i, geom1 in enumerate(geometries):
            for geom2 in geometries[i+1:]:
                pair_key = (geom1, geom2)
                
                # Compute multiple divergence metrics for this pair
                kl_div = self._compute_kl_divergence(
                    geometric_outputs[geom1],
                    geometric_outputs[geom2]
                )
                js_div = self._compute_js_divergence(
                    geometric_outputs[geom1],
                    geometric_outputs[geom2]
                )
                rank_corr = self._compute_rank_correlation(
                    geometric_outputs[geom1],
                    geometric_outputs[geom2]
                )
                
                # Combine metrics into single divergence score
                # Higher KL/JS = more divergent
                # Lower correlation = more divergent
                combined_divergence = (kl_div + js_div) / 2 + (1 - abs(rank_corr))
                
                pairwise[pair_key] = combined_divergence
        
        return pairwise
    
    def identify_conflicting_geometries(self, geometric_outputs: Dict[str, torch.Tensor], 
                                       threshold: float = 0.7) -> List[Tuple[str, str]]:
        """
        Identify pairs of geometries with high interpretive conflict.
        
        Args:
            geometric_outputs: Dictionary of geometric projections
            threshold: Divergence threshold for conflict detection
            
        Returns:
            List of conflicting geometry pairs
        """
        pairwise_div = self.compute_pairwise_divergences(geometric_outputs)
        
        conflicts = []
        for (geom1, geom2), divergence in pairwise_div.items():
            if divergence > threshold:
                conflicts.append((geom1, geom2))
        
        return conflicts