# Data Directory

This directory contains data files and results from geometric analysis experiments.

## Structure

- `phase_correlations.json` - Links to empirical findings from Ouroboros analysis
- `cache/` - Cached model outputs (gitignored)
- `processed/` - Processed attention patterns (gitignored)

## Phase Correlations

The phase correlations file documents the empirical relationship between Ouroboros phases and geometric preferences:

```json
{
  "analysis": {
    "geometry": "square",
    "percentage": 0.452,
    "p_value": 0.0001
  },
  "synthesis": {
    "geometry": "triangular", 
    "percentage": 0.271,
    "p_value": 0.0001
  },
  "reflection": {
    "geometry": "hexagonal",
    "percentage": 0.180,
    "p_value": 0.0001
  },
  "transformation": {
    "geometry": "pentagonal",
    "percentage": 0.097,
    "p_value": 0.0001
  }
}
```

## Adding Your Own Data

To analyze your own model:

1. Extract attention weights from your model
2. Save as `.pt` or `.npy` files in this directory
3. Run analysis scripts from `experiments/`

## Data Sources

- GPT-3.5 analysis: 1,000 responses analyzed
- Statistical significance: p < 0.0001 for all phase-geometry correlations
- Raw data available upon request (too large for repository)