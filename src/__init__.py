"""
Relativistic Interpretability: A Geometric Framework for Understanding Neural Network Reasoning

This framework shows that neural network interpretations are relative to geometric reference frames,
revealing fundamental constraints in AI reasoning capabilities.
"""

from .geometric_projection import GeometricProjection
from .divergence_metrics import GeometricDivergenceMetrics

__version__ = "1.0.0"
__author__ = "Hillary Danan"
__email__ = "hillarydanan@gmail.com"

__all__ = [
    "GeometricProjection",
    "GeometricDivergenceMetrics",
    "analyze_geometric_reasoning",
]


def analyze_geometric_reasoning(model, inputs, return_raw=False):
    """
    High-level API for geometric analysis of model reasoning.
    
    Args:
        model: PyTorch model with attention mechanism
        inputs: Input tensor or tokens
        return_raw: Whether to return raw projections (default: False)
    
    Returns:
        Analysis results including dominant geometry, entropy, and metrics
    """
    import torch
    
    # Extract attention weights (model-specific, this is a placeholder)
    # In practice, this would hook into the model's attention layers
    with torch.no_grad():
        # This is a simplified example - real implementation would extract actual attention
        if hasattr(model, 'get_attention_weights'):
            attention = model.get_attention_weights(inputs)
        else:
            # Placeholder for demonstration
            seq_len = inputs.shape[-1] if len(inputs.shape) > 1 else 64
            attention = torch.rand(1, 8, seq_len, seq_len)
            attention = torch.softmax(attention, dim=-1)
    
    # Initialize analyzers
    seq_len = attention.shape[-1]
    projector = GeometricProjection(seq_len)
    metrics_calc = GeometricDivergenceMetrics()
    
    # Project onto all geometries
    geometries = ['square', 'triangular', 'hexagonal', 'pentagonal']
    projections = {}
    affinities = {}
    
    for geometry in geometries:
        projections[geometry] = projector.project(attention, geometry)
        affinities[geometry] = projector.measure_affinity(attention, geometry)
    
    # Compute metrics
    dominant_geometry, _ = projector.identify_dominant_geometry(attention)
    geometric_entropy = projector.compute_geometric_entropy(affinities)
    
    # Get filtered projections for divergence
    filtered_projections = {g: p['filtered'] for g, p in projections.items()}
    divergence_metrics = metrics_calc.compute_divergence(filtered_projections)
    
    # Identify transformation bottleneck
    pentagonal_capacity = affinities['pentagonal']
    bottleneck_severity = 1.0 - pentagonal_capacity  # How restricted is creative thinking?
    
    # Package results
    results = {
        'dominant_geometry': dominant_geometry,
        'affinities': affinities,
        'entropy': geometric_entropy,
        'divergence': divergence_metrics,
        'pentagonal_capacity': pentagonal_capacity,
        'bottleneck_severity': bottleneck_severity,
    }
    
    if return_raw:
        results['raw_projections'] = projections
    
    return AnalysisResults(**results)


class AnalysisResults:
    """Container for geometric analysis results."""
    
    def __init__(self, dominant_geometry, affinities, entropy, divergence, 
                 pentagonal_capacity, bottleneck_severity, raw_projections=None):
        self.dominant_geometry = dominant_geometry
        self.affinities = affinities
        self.entropy = entropy
        self.divergence = divergence
        self.pentagonal_capacity = pentagonal_capacity
        self.bottleneck_severity = bottleneck_severity
        self.raw_projections = raw_projections
    
    def __repr__(self):
        return (f"AnalysisResults(dominant='{self.dominant_geometry}', "
                f"entropy={self.entropy:.3f}, "
                f"pentagonal_capacity={self.pentagonal_capacity:.1%})")
    
    def summary(self):
        """Print human-readable summary of analysis."""
        print("="*60)
        print("GEOMETRIC REASONING ANALYSIS")
        print("="*60)
        print(f"Dominant Geometry: {self.dominant_geometry.upper()}")
        print(f"Geometric Entropy: {self.entropy:.3f}")
        print(f"Transformation Capacity: {self.pentagonal_capacity:.1%}")
        print(f"Bottleneck Severity: {self.bottleneck_severity:.1%}")
        print("\nGeometric Affinities:")
        for geom, affinity in self.affinities.items():
            bar = "█" * int(affinity * 20) + "░" * (20 - int(affinity * 20))
            print(f"  {geom:12s} {bar} {affinity:.3f}")
        print("="*60)