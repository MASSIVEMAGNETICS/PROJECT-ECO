"""
VICTOR Integration Tests
End-to-end tests for the Victor system.
"""

import asyncio
import numpy as np
import pytest

from victor_integration import (
    DeploymentConfig,
    TransformerOutput,
    VictorSystem,
)


@pytest.mark.asyncio
async def test_full_generation_pipeline(tmp_path):
    """Test complete generation pipeline."""
    system = VictorSystem(
        node_id="integration-test",
        config=DeploymentConfig.dev(),
        checkpoint_dir=str(tmp_path / "checkpoints")
    )
    
    await system.initialize()
    
    try:
        # Run multiple generations
        for i in range(10):
            embeddings = np.random.randn(16, 768).astype(np.float32)
            attention = np.random.rand(8, 16, 16).astype(np.float32)
            attention = attention / attention.sum(axis=-1, keepdims=True)
            
            output = TransformerOutput(
                embeddings=embeddings,
                attention_weights=attention,
                metadata={"iteration": i}
            )
            
            result = await system.generate(output)
            
            # Verify result structure
            assert result["generation_id"] == i + 1
            assert 0 <= result["consciousness_state"]["awareness_level"] <= 1
            assert result["health"] in ["healthy", "anomalous"]
        
        # Verify system status
        status = await system.get_system_status()
        assert status["generation_count"] == 10
        assert status["initialized"] is True
        
    finally:
        await system.shutdown()


@pytest.mark.asyncio
async def test_checkpoint_continuity(tmp_path):
    """Test system continuity through checkpoints."""
    checkpoint_dir = str(tmp_path / "checkpoints")
    
    # First session
    system1 = VictorSystem(
        node_id="continuity-test",
        config=DeploymentConfig.dev(),
        checkpoint_dir=checkpoint_dir
    )
    system1.config.checkpoint_interval = 5  # Low for testing
    
    await system1.initialize()
    
    # Generate some data
    for i in range(10):
        output = TransformerOutput(
            embeddings=np.random.randn(16, 768).astype(np.float32)
        )
        await system1.generate(output)
    
    await system1.shutdown()
    
    # Second session should restore from checkpoint
    system2 = VictorSystem(
        node_id="continuity-test",
        config=DeploymentConfig.dev(),
        checkpoint_dir=checkpoint_dir
    )
    
    await system2.initialize()
    
    # Verify checkpoints exist
    checkpoints = system2.checkpoint_manager.list_checkpoints()
    assert len(checkpoints) >= 1
    
    await system2.shutdown()
