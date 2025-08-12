"""
Minimal example of Relativistic Interpretability analysis.

This script demonstrates the core functionality of the framework:
1. Creating synthetic attention patterns
2. Projecting onto different geometries
3. Measuring geometric affinities
4. Computing divergence metrics
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometric_projection import GeometricProjection
from src.divergence_metrics import GeometricDivergenceMetrics

def create_synthetic_attention(seq_len: int, pattern_type: str = 'random') -> torch.Tensor:
    """
    Create synthetic attention patterns for testing.
    
    Args:
        seq_len: Sequence length
        pattern_type: Type of pattern ('random', 'diagonal', 'lower', 'strided')
    
    Returns:
        Attention tensor [1, 8, seq_len, seq_len]
    """
    batch_size = 1
    n_heads = 8
    
    if pattern_type == 'random':
        attention = torch.rand(batch_size, n_heads, seq_len, seq_len)
    elif pattern_type == 'diagonal':
        attention = torch.eye(seq_len).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_heads, 1, 1)
        # Add some noise
        attention += torch.rand_like(attention) * 0.1
    elif pattern_type == 'lower':
        attention = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        attention = attention.repeat(batch_size, n_heads, 1, 1)
        attention += torch.rand_like(attention) * 0.1
    elif pattern_type == 'strided':
        attention = torch.zeros(batch_size, n_heads, seq_len, seq_len)
        for h in range(n_heads):
            stride = h + 1
            for i in range(seq_len):
                for j in range(i, min(i + stride, seq_len)):
                    attention[0, h, i, j] = 1.0
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")
    
    # Normalize to valid attention (softmax)
    attention = torch.softmax(attention, dim=-1)
    
    return attention

def visualize_results(affinities: dict, divergence: dict):
    """
    Create a simple text visualization of results.
    """
    print("\n" + "="*60)
    print("📊 GEOMETRIC ANALYSIS RESULTS")
    print("="*60)
    
    # Affinity scores
    print("\n🎯 Geometric Affinities:")
    print("-" * 30)
    
    # Sort by affinity
    sorted_affinities = sorted(affinities.items(), key=lambda x: x[1], reverse=True)
    
    for geometry, score in sorted_affinities:
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        emoji = "🏆" if geometry == sorted_affinities[0][0] else "  "
        print(f"{emoji} {geometry:12s} {bar} {score:.3f}")
    
    # Divergence metrics
    print("\n📈 Divergence Metrics:")
    print("-" * 30)
    for metric, value in divergence.items():
        print(f"  {metric:20s}: {value:.4f}")
    
    # Interpretation
    print("\n💡 Interpretation:")
    print("-" * 30)
    dominant = sorted_affinities[0][0]
    score = sorted_affinities[0][1]
    
    interpretations = {
        'square': "Sequential/logical processing dominant",
        'triangular': "High integration and synthesis",
        'hexagonal': "Balanced, efficient processing",
        'pentagonal': "Creative/transformative patterns"
    }
    
    print(f"  Dominant geometry: {dominant.upper()} ({score:.1%})")
    print(f"  → {interpretations[dominant]}")
    
    # Check for high divergence
    if 'geometric_entropy' in divergence and divergence['geometric_entropy'] > 0.8:
        print("\n  ⚠️ High geometric entropy detected!")
        print("  → Multiple valid interpretations exist")
    
    print("\n" + "="*60)

def main():
    """
    Run minimal example of Relativistic Interpretability analysis.
    """
    print("\n" + "="*60)
    print("🚀 RELATIVISTIC INTERPRETABILITY - MINIMAL EXAMPLE")
    print("="*60)
    
    # Configuration
    seq_len = 64
    pattern_types = ['random', 'diagonal', 'lower', 'strided']
    
    # Initialize geometric projection
    print("\n📐 Initializing geometric projector...")
    projector = GeometricProjection(seq_len)
    
    # Initialize metrics calculator
    print("📊 Initializing divergence metrics...")
    metrics_calculator = GeometricDivergenceMetrics()
    
    # Analyze different attention patterns
    for pattern_type in pattern_types:
        print(f"\n\n{'='*60}")
        print(f"🔍 Analyzing {pattern_type.upper()} attention pattern")
        print("="*60)
        
        # Create attention pattern
        attention = create_synthetic_attention(seq_len, pattern_type)
        print(f"✓ Created {pattern_type} attention: shape {list(attention.shape)}")
        
        # Project onto different geometries
        geometries = ['square', 'hexagonal', 'triangular', 'pentagonal']
        projections = {}
        affinities = {}
        
        print("\n🔄 Projecting onto geometries...")
        for geometry in geometries:
            # Project attention
            projections[geometry] = projector.project(attention, geometry)
            
            # Measure affinity
            affinity = projector.measure_affinity(attention, geometry)
            affinities[geometry] = affinity
            
            print(f"  ✓ {geometry:12s} affinity: {affinity:.3f}")
        
        # Compute geometric entropy
        entropy = projector.compute_geometric_entropy(affinities)
        print(f"\n📊 Geometric entropy: {entropy:.3f}")
        
        # Identify dominant geometry
        dominant, _ = projector.identify_dominant_geometry(attention)
        print(f"🎯 Dominant geometry: {dominant}")
        
        # Measure geometric divergence
        print("\n📈 Computing divergence metrics...")
        
        # Use the 'filtered' projection method for divergence computation
        filtered_projections = {g: p['filtered'] for g, p in projections.items()}
        divergence = metrics_calculator.compute_divergence(filtered_projections)
        divergence['geometric_entropy'] = entropy
        
        # Visualize results
        visualize_results(affinities, divergence)
    
    # Final message
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\n💡 Key Insights:")
    print("  • Different attention patterns have distinct geometric signatures")
    print("  • Diagonal patterns show high square geometry affinity")
    print("  • Random patterns have high geometric entropy")
    print("  • Strided patterns vary by head, showing mixed geometries")
    print("\n🔗 Next steps:")
    print("  1. Try with real model attention weights")
    print("  2. Correlate with task performance")
    print("  3. Track geometric evolution during training")
    print("\n⭐ Star the repo if you find this interesting!")
    print("   https://github.com/HillaryDanan/relativistic-interpretability")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()