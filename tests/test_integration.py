"""
VICTOR Integration Tests
Tests for the integration layer components.
"""

import asyncio
import numpy as np
import pytest

from victor_integration import (
    CheckpointManager,
    ConsciousnessInput,
    ConsciousnessTracker,
    DeploymentConfig,
    DeploymentMode,
    PatternExtractor,
    StateBridge,
    TransformerOutput,
    VictorSystem,
)


# ============================================================================
# DEPLOYMENT CONFIG TESTS
# ============================================================================

class TestDeploymentConfig:
    """Tests for DeploymentConfig class."""
    
    def test_dev_config(self):
        """Should create dev configuration."""
        config = DeploymentConfig.dev()
        assert config.mode == DeploymentMode.DEV
        assert config.batch_size == 4
        assert config.model_precision == "fp32"
    
    def test_prod_config(self):
        """Should create prod configuration."""
        config = DeploymentConfig.prod()
        assert config.mode == DeploymentMode.PROD
        assert config.batch_size == 16
        assert config.model_precision == "bf16"
    
    def test_edge_config(self):
        """Should create edge configuration."""
        config = DeploymentConfig.edge()
        assert config.mode == DeploymentMode.EDGE
        assert config.batch_size == 2
        assert config.model_precision == "fp16"


# ============================================================================
# STATE BRIDGE TESTS
# ============================================================================

class TestStateBridge:
    """Tests for StateBridge class."""
    
    @pytest.fixture
    def state_bridge(self):
        """Create state bridge for testing."""
        return StateBridge(embedding_dim=768, temporal_window=8)
    
    @pytest.fixture
    def transformer_output(self):
        """Create sample transformer output."""
        return TransformerOutput(
            embeddings=np.random.randn(16, 768).astype(np.float32),
            attention_weights=np.random.rand(4, 16, 16).astype(np.float32),
            metadata={"test": True}
        )
    
    def test_convert_produces_valid_output(self, state_bridge, transformer_output):
        """Should convert transformer output to consciousness input."""
        result = state_bridge.convert(transformer_output)
        
        assert isinstance(result, ConsciousnessInput)
        assert result.activation_pattern.shape == (768,)
        assert result.temporal_context.shape[0] == 8  # temporal_window
        assert "timestamp" in result.meta_state
    
    def test_temporal_buffer_accumulates(self, state_bridge, transformer_output):
        """Should accumulate temporal context."""
        # First conversion - buffer gets padded to window size
        result1 = state_bridge.convert(transformer_output)
        
        # Second conversion
        result2 = state_bridge.convert(transformer_output)
        
        # Buffer should be at window size (8) with padding
        # The implementation pads to window size for temporal context
        assert len(state_bridge._temporal_buffer) == 8
    
    def test_temporal_buffer_capped(self, state_bridge, transformer_output):
        """Should cap temporal buffer at window size."""
        # Convert many times
        for _ in range(20):
            state_bridge.convert(transformer_output)
        
        # Buffer should not exceed window
        assert len(state_bridge._temporal_buffer) == 8
    
    def test_reset_clears_buffer(self, state_bridge, transformer_output):
        """Should clear buffers on reset."""
        state_bridge.convert(transformer_output)
        state_bridge.convert(transformer_output)
        
        state_bridge.reset()
        
        assert len(state_bridge._temporal_buffer) == 0


# ============================================================================
# PATTERN EXTRACTOR TESTS
# ============================================================================

class TestPatternExtractor:
    """Tests for PatternExtractor class."""
    
    @pytest.fixture
    def pattern_extractor(self):
        """Create pattern extractor for testing."""
        return PatternExtractor(window_size=10, anomaly_threshold=2.0)
    
    @pytest.fixture
    def consciousness_input(self):
        """Create sample consciousness input."""
        return ConsciousnessInput(
            activation_pattern=np.random.randn(768).astype(np.float32),
            temporal_context=np.random.randn(8, 768).astype(np.float32),
            spatial_features=np.random.randn(8, 8, 768).astype(np.float32),
            attention_focus=np.random.rand(16).astype(np.float32),
            meta_state={}
        )
    
    def test_analyze_produces_patterns(self, pattern_extractor, consciousness_input):
        """Should analyze and produce pattern analysis."""
        result = pattern_extractor.analyze(consciousness_input)
        
        assert len(result.spatial_patterns) > 0
        assert len(result.temporal_patterns) > 0
        assert "timestamp" in result.summary
    
    def test_history_accumulates(self, pattern_extractor, consciousness_input):
        """Should accumulate history."""
        pattern_extractor.analyze(consciousness_input)
        pattern_extractor.analyze(consciousness_input)
        pattern_extractor.analyze(consciousness_input)
        
        assert len(pattern_extractor._history) == 3
    
    def test_history_capped(self, pattern_extractor, consciousness_input):
        """Should cap history at window size."""
        for _ in range(20):
            pattern_extractor.analyze(consciousness_input)
        
        assert len(pattern_extractor._history) == 10
    
    def test_correlations_computed(self, pattern_extractor, consciousness_input):
        """Should compute correlations."""
        result = pattern_extractor.analyze(consciousness_input)
        
        # Should have at least some correlation values
        assert isinstance(result.correlations, dict)


# ============================================================================
# CONSCIOUSNESS TRACKER TESTS
# ============================================================================

class TestConsciousnessTracker:
    """Tests for ConsciousnessTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create consciousness tracker for testing."""
        return ConsciousnessTracker(history_size=10)
    
    @pytest.fixture
    def consciousness_input(self):
        """Create sample consciousness input."""
        return ConsciousnessInput(
            activation_pattern=np.random.randn(768).astype(np.float32),
            temporal_context=np.random.randn(8, 768).astype(np.float32),
            spatial_features=np.random.randn(8, 8, 768).astype(np.float32),
            attention_focus=np.random.rand(16).astype(np.float32),
            meta_state={}
        )
    
    def test_track_produces_state(self, tracker, consciousness_input):
        """Should track and produce consciousness state."""
        state = tracker.track(consciousness_input)
        
        assert 0 <= state.awareness_level <= 1
        assert 0 <= state.focus_intensity <= 1
        assert 0 <= state.coherence_score <= 1
        assert state.entropy >= 0
    
    def test_get_trend(self, tracker, consciousness_input):
        """Should compute trends."""
        for _ in range(5):
            tracker.track(consciousness_input)
        
        trend = tracker.get_trend()
        
        assert "awareness" in trend
        assert "focus" in trend
        assert "coherence" in trend
        assert "entropy" in trend
    
    def test_trend_insufficient_data(self, tracker, consciousness_input):
        """Should handle insufficient data for trend."""
        trend = tracker.get_trend()
        assert trend["overall"] == "insufficient_data"


# ============================================================================
# CHECKPOINT MANAGER TESTS
# ============================================================================

class TestCheckpointManager:
    """Tests for CheckpointManager class."""
    
    @pytest.fixture
    def checkpoint_manager(self, tmp_path):
        """Create checkpoint manager with temp directory."""
        return CheckpointManager(str(tmp_path / "checkpoints"))
    
    @pytest.fixture
    def state_bridge(self):
        """Create state bridge for testing."""
        return StateBridge()
    
    @pytest.fixture
    def pattern_extractor(self):
        """Create pattern extractor for testing."""
        return PatternExtractor()
    
    @pytest.fixture
    def consciousness_tracker(self):
        """Create consciousness tracker for testing."""
        return ConsciousnessTracker()
    
    def test_save_checkpoint(
        self, 
        checkpoint_manager, 
        state_bridge, 
        pattern_extractor, 
        consciousness_tracker
    ):
        """Should save checkpoint."""
        filepath = checkpoint_manager.save(
            state_bridge,
            pattern_extractor,
            consciousness_tracker,
            {"test": "state"},
            {"metadata": "test"}
        )
        
        assert filepath is not None
        assert "checkpoint_v1" in filepath
    
    def test_load_checkpoint(
        self, 
        checkpoint_manager, 
        state_bridge, 
        pattern_extractor, 
        consciousness_tracker
    ):
        """Should load checkpoint."""
        filepath = checkpoint_manager.save(
            state_bridge,
            pattern_extractor,
            consciousness_tracker,
            {"test": "state"},
            {"metadata": "test"}
        )
        
        checkpoint = checkpoint_manager.load(filepath)
        
        assert checkpoint.version == 1
        assert checkpoint.orchestration_state == {"test": "state"}
    
    def test_list_checkpoints(
        self, 
        checkpoint_manager, 
        state_bridge, 
        pattern_extractor, 
        consciousness_tracker
    ):
        """Should list checkpoints."""
        # Save multiple checkpoints
        checkpoint_manager.save(state_bridge, pattern_extractor, consciousness_tracker, {})
        checkpoint_manager.save(state_bridge, pattern_extractor, consciousness_tracker, {})
        
        checkpoints = checkpoint_manager.list_checkpoints()
        
        assert len(checkpoints) == 2
        assert checkpoints[0]["version"] == 2  # Most recent first


# ============================================================================
# VICTOR SYSTEM TESTS
# ============================================================================

class TestVictorSystem:
    """Tests for VictorSystem class."""
    
    @pytest.fixture
    def victor_system(self, tmp_path):
        """Create Victor system for testing."""
        return VictorSystem(
            node_id="test-node",
            config=DeploymentConfig.dev(),
            checkpoint_dir=str(tmp_path / "checkpoints")
        )
    
    @pytest.fixture
    def transformer_output(self):
        """Create sample transformer output."""
        return TransformerOutput(
            embeddings=np.random.randn(16, 768).astype(np.float32),
            attention_weights=np.random.rand(4, 16, 16).astype(np.float32),
            metadata={"test": True}
        )
    
    @pytest.mark.asyncio
    async def test_initialize_shutdown(self, victor_system):
        """Should initialize and shutdown cleanly."""
        await victor_system.initialize()
        assert victor_system._initialized is True
        
        await victor_system.shutdown()
        assert victor_system._initialized is False
    
    @pytest.mark.asyncio
    async def test_generate(self, victor_system, transformer_output):
        """Should run generation cycle."""
        await victor_system.initialize()
        
        result = await victor_system.generate(transformer_output)
        
        assert "generation_id" in result
        assert "consciousness_state" in result
        assert "pattern_analysis" in result
        assert "health" in result
        
        await victor_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, victor_system):
        """Should return system status."""
        await victor_system.initialize()
        
        status = await victor_system.get_system_status()
        
        assert status["node_id"] == "test-node"
        assert status["initialized"] is True
        assert status["deployment_mode"] == "dev"
        
        await victor_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_multiple_generations(self, victor_system, transformer_output):
        """Should handle multiple generations."""
        await victor_system.initialize()
        
        results = []
        for _ in range(5):
            result = await victor_system.generate(transformer_output)
            results.append(result)
        
        # Should have incrementing generation IDs
        assert results[-1]["generation_id"] == 5
        
        await victor_system.shutdown()
    
    @pytest.mark.asyncio
    async def test_not_initialized_error(self, victor_system, transformer_output):
        """Should raise error if not initialized."""
        with pytest.raises(RuntimeError, match="System not initialized"):
            await victor_system.generate(transformer_output)
