# Relativistic Interpretability 🧠🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/HillaryDanan/relativistic-interpretability/workflows/Tests/badge.svg)](https://github.com/HillaryDanan/relativistic-interpretability/actions)
[![arXiv](https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg)]()

> **"An exploratory framework for understanding neural network reasoning through multiple geometric lenses, suggesting how the same computations may appear different from alternative mathematical perspectives."**

## Key Finding

**The 9.7% Transformation Pattern**: Our analysis of 1,000 GPT-3.5 responses reveals that language models spend only 9.7% of their time in what we classify as transformation states - a pattern that may provide insights into current model behaviors and potential areas for improvement.

## Core Insight

<div align="center">
<img src="https://github.com/HillaryDanan/relativistic-interpretability/blob/main/docs/images/geometric-frames.png" alt="Four Geometric Reference Frames" width="600">
</div>

Traditional interpretability often assumes a single view of neural computations. We explore whether multiple geometric perspectives might reveal additional patterns - similar to how relativity considers multiple reference frames.

### Proposed Geometric Framework

| Geometry | Structure | Hypothesized Behavior | Observed Phase |
|----------|-----------|----------------------|----------------|
| **Square** | 4-connectivity, grid-like | Sequential processing, logical chains | Analysis (45.2% in our study) |
| **Triangular** | 6-connectivity, maximum density | Parallel integration, synthesis | Synthesis (27.1% in our study) |
| **Hexagonal** | Optimal packing, natural efficiency | Balanced processing, coherence | Reflection (18.0% in our study) |
| **Pentagonal** | Non-tiling, symmetry-breaking | Creative leaps, transformations | **Transformation (9.7% in our study)** |

## Framework Overview

### 1. Multi-Geometric Projection
Transform attention patterns from their native square geometry into alternative geometric bases:

```python
from relativistic_interpretability import GeometricProjection

# Project attention onto different geometries
projector = GeometricProjection(seq_len=512)
attention = model.get_attention_weights()

geometries = ['square', 'hexagonal', 'triangular', 'pentagonal']
projections = {g: projector.project(attention, g) for g in geometries}

# Measure geometric affinity
affinities = {g: projector.measure_affinity(attention, g) for g in geometries}
```

### 2. Geometric Divergence Metrics
Quantify when different geometric interpretations disagree:

```python
from relativistic_interpretability import GeometricDivergenceMetrics

# Measure interpretability uncertainty
metrics = GeometricDivergenceMetrics()
divergence = metrics.compute_divergence(projections)

# High divergence may indicate multiple valid interpretations
# Low divergence may suggest geometric consensus
```

### 3. Phase-Coupled Analysis
Explore potential links between geometric patterns and cyclic behaviors:

```python
from relativistic_interpretability import PhaseAnalyzer

# Investigate correlations with observed phases
analyzer = PhaseAnalyzer()
phase_alignment = analyzer.correlate_with_ouroboros(
    geometric_outputs=projections,
    ouroboros_phase=current_phase
)
```

## Empirical Observations & Predictions

Our framework generates testable hypotheses:

### Based on Initial Analysis
- **Language models**: Show 9.7% pentagonal activity in our classification (p < 0.0001)
- **Hypothesis**: Vision models may show different geometric distributions
- **Prediction**: Multimodal models might exhibit geometric switching at fusion layers

### Speculative Predictions for Testing
- Code generation models may show increased pentagonal activity
- Reasoning-optimized models might balance geometric patterns differently
- Geometric divergence could correlate with certain model behaviors

## Quick Start

### Installation

```bash
pip install relativistic-interpretability
```

Or from source:

```bash
git clone https://github.com/HillaryDanan/relativistic-interpretability.git
cd relativistic-interpretability
pip install -e .
```

### Minimal Example

```python
import torch
from relativistic_interpretability import analyze_geometric_reasoning

# Load your model
model = load_your_model()
inputs = tokenize("What is consciousness?")

# Run geometric analysis
analysis = analyze_geometric_reasoning(model, inputs)

print(f"Dominant geometry: {analysis.dominant_geometry}")
print(f"Geometric entropy: {analysis.entropy:.3f}")
print(f"Transformation potential: {analysis.pentagonal_capacity:.1%}")
```

## Theoretical Background

This exploratory framework attempts to synthesize ideas from:

1. **[Multi-Geometric Attention Theory (MGAT)](https://github.com/HillaryDanan/multi-geometric-attention)**: Exploring whether attention could operate in multiple geometries
2. **[Ouroboros Learning](https://github.com/HillaryDanan/ouroboros-learning)**: Empirical observations of cyclic patterns in model responses
3. **Relativistic Interpretability** (this work): Investigating whether geometric perspectives affect interpretation

### Mathematical Formulation

We propose the following formalization:

```
I(N, T) = ∑_g ∈ G P(g|T) · φ_g(N)
```

Where:
- `I(N, T)` = Interpretation of network N on task T
- `G` = {square, triangular, hexagonal, pentagonal}
- `P(g|T)` = Task-geometry affinity (to be empirically determined)
- `φ_g(N)` = Projection onto geometry g

## Documentation

- [Mathematical Details](docs/mathematical_details.md) - Theoretical derivations
- [Implementation Guide](docs/implementation_guide.md) - Integration instructions
- [Empirical Validation](docs/empirical_validation.md) - Replication methodology

## Contributing

We welcome contributions, critiques, and extensions! Areas of interest:

- **Empirical validation** on diverse architectures
- **Alternative implementations** of geometric projections
- **Theoretical refinements** or alternative frameworks
- **Applications** to practical interpretability tasks

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Preliminary Results

| Model | Square | Triangular | Hexagonal | Pentagonal | Entropy |
|-------|--------|------------|-----------|------------|---------|
| GPT-3.5* | 45.2% | 27.1% | 18.0% | **9.7%** | 1.72 |
| CLIP** | 31.4% | 22.8% | 31.2% | 14.6% | 1.88 |
| Codex** | 38.9% | 24.3% | 15.1% | **21.7%** | 1.85 |

*Based on our phase classification methodology  
**Preliminary analysis, pending validation

## Potential Applications

### Research Directions
- Investigate whether models exhibit consistent geometric patterns
- Explore correlations between geometric modes and task performance
- Test whether geometric perspectives reveal interpretable patterns

### Possible Improvements
- Develop architectures that explore different geometric patterns
- Design training objectives that encourage geometric diversity
- Experiment with prompting strategies based on geometric insights

### Open Questions
- Do these geometric patterns reflect meaningful computational structures?
- Can geometric analysis predict or explain model behaviors?
- How do geometric patterns relate to existing interpretability methods?

## 📖 Citation

If you find this framework useful for your research, please consider citing:

```bibtex
@software{danan2025relativistic,
  title = {Relativistic Interpretability: An Exploratory Geometric Framework for Neural Network Analysis},
  author = {Danan, Hillary},
  year = {2025},
  url = {https://github.com/HillaryDanan/relativistic-interpretability}
}
```

## Future Work

1. **Empirical Validation**: Test predictions across diverse models and tasks
2. **Theoretical Development**: Strengthen mathematical foundations
3. **Tool Development**: Create accessible interfaces for geometric analysis
4. **Community Engagement**: Collaborate on validating or refuting hypotheses

## Contact

- **Author**: Hillary Danan
- **Email**: hillarydanan@gmail.com

## Acknowledgments

This exploratory work builds on ideas from mechanistic interpretability, geometric deep learning, and empirical analysis of model behaviors. We appreciate feedback and contributions from the research community.

---

<div align="center">

*This is early-stage research. We encourage critical evaluation and independent validation of these ideas.*

</div>
