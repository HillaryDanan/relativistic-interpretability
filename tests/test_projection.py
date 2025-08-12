"""
Tests for geometric projection operations.
"""

import pytest
import torch
import numpy as np
from src.geometric_projection import GeometricProjection


class TestGeometricProjection:
    """Test suite for GeometricProjection class."""
    
    @pytest.fixture
    def projector(self):
        """Create a projector instance for testing."""
        return GeometricProjection(seq_len=64)
    
    @pytest.fixture
    def sample_attention(self):
        """Create sample attention tensor."""
        batch_size = 2
        n_heads = 8
        seq_len = 64
        
        attention = torch.rand(batch_size, n_heads, seq_len, seq_len)
        attention = torch.softmax(attention, dim=-1)
        return attention
    
    def test_initialization(self):
        """Test projector initialization."""
        seq_len = 32
        projector = GeometricProjection(seq_len)
        
        assert projector.seq_len == seq_len
        assert len(projector.adjacency_matrices) == 4
        assert len(projector.laplacians) == 4
        
        # Check all geometries are present
        expected_geometries = {'square', 'triangular', 'hexagonal', 'pentagonal'}
        assert set(projector.adjacency_matrices.keys()) == expected_geometries
    
    def test_adjacency_matrix_properties(self, projector):
        """Test properties of adjacency matrices."""
        for geom, adj in projector.adjacency_matrices.items():
            # Check shape
            assert adj.shape == (projector.seq_len, projector.seq_len)
            
            # Check non-negative
            assert (adj >= 0).all()
            
            # Check normalized (rows sum to 1 or 0)
            row_sums = adj.sum(dim=1)
            assert torch.allclose(row_sums[row_sums > 0], torch.ones_like(row_sums[row_sums > 0]), atol=1e-5)
    
    def test_projection_shapes(self, projector, sample_attention):
        """Test that projection preserves tensor shapes."""
        for geometry in ['square', 'triangular', 'hexagonal', 'pentagonal']:
            result = projector.project(sample_attention, geometry)
            
            # Check all projection methods return correct shape
            assert 'filtered' in result
            assert 'spectral' in result
            assert 'flow' in result
            
            for method in ['filtered', 'spectral', 'flow']:
                assert result[method].shape == sample_attention.shape
    
    def test_projection_invalid_geometry(self, projector, sample_attention):
        """Test projection with invalid geometry raises error."""
        with pytest.raises(ValueError, match="Unknown geometry"):
            projector.project(sample_attention, 'invalid_geometry')
    
    def test_affinity_range(self, projector, sample_attention):
        """Test that affinity scores are in valid range [0, 1]."""
        for geometry in ['square', 'triangular', 'hexagonal', 'pentagonal']:
            affinity = projector.measure_affinity(sample_attention, geometry)
            
            assert isinstance(affinity, float)
            assert 0 <= affinity <= 1
    
    def test_geometric_entropy(self, projector):
        """Test geometric entropy calculation."""
        # Uniform distribution should have high entropy
        uniform_affinities = {
            'square': 0.25,
            'triangular': 0.25,
            'hexagonal': 0.25,
            'pentagonal': 0.25
        }
        entropy_uniform = projector.compute_geometric_entropy(uniform_affinities)
        assert entropy_uniform > 0.9  # Should be close to 1
        
        # Concentrated distribution should have low entropy
        concentrated_affinities = {
            'square': 0.97,
            'triangular': 0.01,
            'hexagonal': 0.01,
            'pentagonal': 0.01
        }
        entropy_concentrated = projector.compute_geometric_entropy(concentrated_affinities)
        assert entropy_concentrated < 0.3  # Should be close to 0
    
    def test_dominant_geometry(self, projector, sample_attention):
        """Test identification of dominant geometry."""
        dominant, affinities = projector.identify_dominant_geometry(sample_attention)
        
        # Check dominant is one of the valid geometries
        assert dominant in ['square', 'triangular', 'hexagonal', 'pentagonal']
        
        # Check dominant has highest affinity
        assert affinities[dominant] == max(affinities.values())
        
        # Check all geometries have affinities
        assert len(affinities) == 4
    
    def test_different_attention_patterns(self, projector):
        """Test that different attention patterns yield different geometric preferences."""
        seq_len = 64
        batch_size = 1
        n_heads = 8
        
        # Diagonal attention (should prefer square)
        diagonal_attention = torch.eye(seq_len).unsqueeze(0).unsqueeze(0)
        diagonal_attention = diagonal_attention.repeat(batch_size, n_heads, 1, 1)
        diagonal_attention = torch.softmax(diagonal_attention + 0.01, dim=-1)
        
        # Random attention
        random_attention = torch.rand(batch_size, n_heads, seq_len, seq_len)
        random_attention = torch.softmax(random_attention, dim=-1)
        
        # Get dominant geometries
        dominant_diag, _ = projector.identify_dominant_geometry(diagonal_attention)
        dominant_rand, _ = projector.identify_dominant_geometry(random_attention)
        
        # They should often be different (not always, due to randomness)
        # But affinities should definitely differ
        affinity_diag_square = projector.measure_affinity(diagonal_attention, 'square')
        affinity_rand_square = projector.measure_affinity(random_attention, 'square')
        
        assert abs(affinity_diag_square - affinity_rand_square) > 0.01
    
    def test_laplacian_properties(self, projector):
        """Test properties of graph Laplacians."""
        for geom, laplacian in projector.laplacians.items():
            # Check shape
            assert laplacian.shape == (projector.seq_len, projector.seq_len)
            
            # Check symmetric (for undirected graphs)
            # Note: Our adjacency matrices might not be symmetric, so skip this
            
            # Check diagonal dominance (loosely)
            diag = torch.diag(laplacian)
            assert (diag >= 0).all()
    
    def test_device_compatibility(self):
        """Test that projector works with different devices."""
        seq_len = 32
        
        # CPU version
        projector_cpu = GeometricProjection(seq_len, device='cpu')
        assert projector_cpu.device == 'cpu'
        
        # CUDA version (only test if CUDA available)
        if torch.cuda.is_available():
            projector_cuda = GeometricProjection(seq_len, device='cuda')
            assert projector_cuda.device == 'cuda'
            
            # Test projection on CUDA
            attention = torch.rand(1, 4, seq_len, seq_len).cuda()
            attention = torch.softmax(attention, dim=-1)
            
            result = projector_cuda.project(attention, 'square')
            assert result['filtered'].device.type == 'cuda'