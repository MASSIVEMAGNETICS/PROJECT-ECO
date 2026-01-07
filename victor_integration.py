"""
VICTOR Integration Layer
Unified system combining all intelligence components.

This module provides:
- StateBridge: Converts transformer outputs → consciousness inputs
- PatternExtractor: Analyzes spatial and temporal patterns
- VictorSystem: Unified system combining all components
- Full generation cycle with consciousness tracking
- Checkpoint saving/loading for continuity
- Deployment configs for dev/prod/edge
"""

import asyncio
import json
import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from victor_infrastructure import (
    OrchestrationEngine,
    create_orchestration_engine,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("victor.integration")


# ============================================================================
# DEPLOYMENT CONFIGURATIONS
# ============================================================================

class DeploymentMode(Enum):
    """Deployment mode configuration."""
    DEV = "dev"
    PROD = "prod"
    EDGE = "edge"


@dataclass
class DeploymentConfig:
    """Configuration for different deployment modes."""
    mode: DeploymentMode
    gpu_type: str
    memory_gb: int
    batch_size: int
    model_precision: str
    enable_monitoring: bool
    checkpoint_interval: int
    max_workers: int
    
    @classmethod
    def dev(cls) -> "DeploymentConfig":
        """Development configuration."""
        return cls(
            mode=DeploymentMode.DEV,
            gpu_type="RTX_4090",
            memory_gb=32,
            batch_size=4,
            model_precision="fp32",
            enable_monitoring=True,
            checkpoint_interval=100,
            max_workers=2
        )
    
    @classmethod
    def prod(cls) -> "DeploymentConfig":
        """Production configuration."""
        return cls(
            mode=DeploymentMode.PROD,
            gpu_type="A100",
            memory_gb=64,
            batch_size=16,
            model_precision="bf16",
            enable_monitoring=True,
            checkpoint_interval=50,
            max_workers=8
        )
    
    @classmethod
    def edge(cls) -> "DeploymentConfig":
        """Edge deployment configuration."""
        return cls(
            mode=DeploymentMode.EDGE,
            gpu_type="T4",
            memory_gb=16,
            batch_size=2,
            model_precision="fp16",
            enable_monitoring=False,
            checkpoint_interval=200,
            max_workers=1
        )


# ============================================================================
# STATE BRIDGE
# ============================================================================

@dataclass
class TransformerOutput:
    """Output from transformer models."""
    embeddings: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    hidden_states: Optional[List[np.ndarray]] = None
    logits: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsciousnessInput:
    """Input format for consciousness boundary."""
    activation_pattern: np.ndarray
    temporal_context: np.ndarray
    spatial_features: np.ndarray
    attention_focus: Optional[np.ndarray] = None
    meta_state: Dict[str, Any] = field(default_factory=dict)


class StateBridge:
    """
    Converts transformer outputs to consciousness inputs.
    
    Handles the translation between:
    - Raw model outputs (embeddings, attention, hidden states)
    - Structured inputs for consciousness/meta-cognition layer
    """
    
    def __init__(
        self,
        embedding_dim: int = 768,
        temporal_window: int = 32,
        spatial_grid: Tuple[int, int] = (8, 8)
    ):
        self.embedding_dim = embedding_dim
        self.temporal_window = temporal_window
        self.spatial_grid = spatial_grid
        
        # State buffers
        self._temporal_buffer: List[np.ndarray] = []
        self._activation_history: List[np.ndarray] = []
    
    def _extract_activation_pattern(
        self,
        output: TransformerOutput
    ) -> np.ndarray:
        """Extract activation pattern from embeddings."""
        embeddings = output.embeddings
        
        # Normalize embeddings
        norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        norm = np.where(norm == 0, 1, norm)
        normalized = embeddings / norm
        
        # Compute activation strengths
        activation = np.mean(normalized, axis=0)
        
        return activation
    
    def _build_temporal_context(
        self,
        output: TransformerOutput
    ) -> np.ndarray:
        """Build temporal context from buffer."""
        # Add current to buffer
        current = self._extract_activation_pattern(output)
        self._temporal_buffer.append(current)
        
        # Maintain window size
        if len(self._temporal_buffer) > self.temporal_window:
            self._temporal_buffer.pop(0)
        
        # Pad if necessary
        while len(self._temporal_buffer) < self.temporal_window:
            self._temporal_buffer.insert(0, np.zeros_like(current))
        
        return np.stack(self._temporal_buffer)
    
    def _extract_spatial_features(
        self,
        output: TransformerOutput
    ) -> np.ndarray:
        """Extract spatial features from attention."""
        if output.attention_weights is None:
            # Default spatial features
            return np.zeros((*self.spatial_grid, self.embedding_dim))
        
        attention = output.attention_weights
        
        # Reshape attention to spatial grid
        # This is a simplified version - actual implementation
        # would depend on model architecture
        h, w = self.spatial_grid
        seq_len = attention.shape[-1]
        
        # Average over heads if multi-head
        if len(attention.shape) > 2:
            attention = np.mean(attention, axis=0)
        
        # Resize to spatial grid
        if seq_len >= h * w:
            attention_flat = attention[:h*w, :h*w]
        else:
            attention_flat = np.zeros((h*w, h*w))
            attention_flat[:seq_len, :seq_len] = attention
        
        spatial = attention_flat.reshape(h, w, h * w)
        
        # Pad to embedding dimension
        if spatial.shape[-1] < self.embedding_dim:
            pad_width = [(0, 0), (0, 0), (0, self.embedding_dim - spatial.shape[-1])]
            spatial = np.pad(spatial, pad_width)
        else:
            spatial = spatial[..., :self.embedding_dim]
        
        return spatial
    
    def _compute_attention_focus(
        self,
        output: TransformerOutput
    ) -> Optional[np.ndarray]:
        """Compute attention focus vector."""
        if output.attention_weights is None:
            return None
        
        attention = output.attention_weights
        
        # Compute entropy-weighted focus
        if len(attention.shape) > 2:
            attention = np.mean(attention, axis=0)
        
        # Row-wise softmax already applied in attention
        entropy = -np.sum(attention * np.log(attention + 1e-10), axis=-1)
        focus = 1.0 / (1.0 + entropy)  # High focus = low entropy
        
        return focus
    
    def convert(self, output: TransformerOutput) -> ConsciousnessInput:
        """Convert transformer output to consciousness input."""
        activation = self._extract_activation_pattern(output)
        temporal = self._build_temporal_context(output)
        spatial = self._extract_spatial_features(output)
        focus = self._compute_attention_focus(output)
        
        # Build meta state
        meta_state = {
            "timestamp": datetime.utcnow().isoformat(),
            "activation_norm": float(np.linalg.norm(activation)),
            "temporal_variance": float(np.var(temporal)),
            "source_metadata": output.metadata
        }
        
        return ConsciousnessInput(
            activation_pattern=activation,
            temporal_context=temporal,
            spatial_features=spatial,
            attention_focus=focus,
            meta_state=meta_state
        )
    
    def reset(self) -> None:
        """Reset internal buffers."""
        self._temporal_buffer.clear()
        self._activation_history.clear()


# ============================================================================
# PATTERN EXTRACTOR
# ============================================================================

@dataclass
class PatternAnalysis:
    """Results of pattern analysis."""
    spatial_patterns: List[Dict[str, Any]]
    temporal_patterns: List[Dict[str, Any]]
    correlations: Dict[str, float]
    anomalies: List[Dict[str, Any]]
    summary: Dict[str, Any]


class PatternExtractor:
    """
    Analyzes spatial and temporal patterns in consciousness states.
    
    Provides:
    - Spatial pattern detection (clustering, regions of interest)
    - Temporal pattern detection (trends, periodicity)
    - Cross-correlation analysis
    - Anomaly detection
    """
    
    def __init__(
        self,
        window_size: int = 100,
        anomaly_threshold: float = 2.5
    ):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self._history: List[ConsciousnessInput] = []
    
    def _analyze_spatial(
        self,
        consciousness_input: ConsciousnessInput
    ) -> List[Dict[str, Any]]:
        """Analyze spatial patterns."""
        spatial = consciousness_input.spatial_features
        patterns = []
        
        # Compute spatial statistics
        mean_activation = np.mean(spatial)
        std_activation = np.std(spatial)
        max_activation = np.max(spatial)
        
        # Find regions of interest (above threshold)
        threshold = mean_activation + std_activation
        hot_spots = np.where(np.mean(spatial, axis=-1) > threshold)
        
        for i, (y, x) in enumerate(zip(*hot_spots)):
            patterns.append({
                "type": "hot_spot",
                "location": (int(y), int(x)),
                "intensity": float(np.mean(spatial[y, x])),
                "index": i
            })
        
        # Overall spatial pattern
        patterns.append({
            "type": "summary",
            "mean": float(mean_activation),
            "std": float(std_activation),
            "max": float(max_activation),
            "hot_spot_count": len(hot_spots[0])
        })
        
        return patterns
    
    def _analyze_temporal(
        self,
        consciousness_input: ConsciousnessInput
    ) -> List[Dict[str, Any]]:
        """Analyze temporal patterns."""
        temporal = consciousness_input.temporal_context
        patterns = []
        
        # Compute temporal statistics
        mean_over_time = np.mean(temporal, axis=-1)
        
        # Trend detection (simple linear regression)
        x = np.arange(len(mean_over_time))
        if len(x) > 1:
            coeffs = np.polyfit(x, mean_over_time, 1)
            trend_slope = coeffs[0]
            trend_direction = "increasing" if trend_slope > 0.01 else (
                "decreasing" if trend_slope < -0.01 else "stable"
            )
        else:
            trend_slope = 0.0
            trend_direction = "stable"
        
        patterns.append({
            "type": "trend",
            "slope": float(trend_slope),
            "direction": trend_direction
        })
        
        # Variance analysis
        variance_over_time = np.var(temporal, axis=-1)
        patterns.append({
            "type": "variance",
            "mean_variance": float(np.mean(variance_over_time)),
            "variance_trend": "increasing" if variance_over_time[-1] > variance_over_time[0] else "decreasing"
        })
        
        return patterns
    
    def _compute_correlations(
        self,
        consciousness_input: ConsciousnessInput
    ) -> Dict[str, float]:
        """Compute cross-correlations."""
        correlations = {}
        
        activation = consciousness_input.activation_pattern.flatten()
        temporal_last = consciousness_input.temporal_context[-1].flatten()
        spatial_mean = consciousness_input.spatial_features.mean(axis=(0, 1))
        
        # Activation-temporal correlation
        if len(activation) == len(temporal_last):
            corr = np.corrcoef(activation, temporal_last)[0, 1]
            correlations["activation_temporal"] = float(corr) if not np.isnan(corr) else 0.0
        
        # Activation-spatial correlation
        if len(activation) == len(spatial_mean):
            corr = np.corrcoef(activation, spatial_mean)[0, 1]
            correlations["activation_spatial"] = float(corr) if not np.isnan(corr) else 0.0
        
        return correlations
    
    def _detect_anomalies(
        self,
        consciousness_input: ConsciousnessInput
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in current state."""
        anomalies = []
        
        if len(self._history) < 2:
            return anomalies
        
        # Compute historical statistics
        historical_activations = [
            np.linalg.norm(h.activation_pattern) 
            for h in self._history
        ]
        mean_activation = np.mean(historical_activations)
        std_activation = np.std(historical_activations)
        
        # Check current activation
        current_activation = np.linalg.norm(consciousness_input.activation_pattern)
        if std_activation > 0:
            z_score = (current_activation - mean_activation) / std_activation
            if abs(z_score) > self.anomaly_threshold:
                anomalies.append({
                    "type": "activation_anomaly",
                    "z_score": float(z_score),
                    "value": float(current_activation),
                    "expected": float(mean_activation),
                    "direction": "high" if z_score > 0 else "low"
                })
        
        return anomalies
    
    def analyze(
        self,
        consciousness_input: ConsciousnessInput
    ) -> PatternAnalysis:
        """Perform comprehensive pattern analysis."""
        # Update history
        self._history.append(consciousness_input)
        if len(self._history) > self.window_size:
            self._history.pop(0)
        
        # Analyze patterns
        spatial_patterns = self._analyze_spatial(consciousness_input)
        temporal_patterns = self._analyze_temporal(consciousness_input)
        correlations = self._compute_correlations(consciousness_input)
        anomalies = self._detect_anomalies(consciousness_input)
        
        # Build summary
        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "history_length": len(self._history),
            "pattern_count": len(spatial_patterns) + len(temporal_patterns),
            "anomaly_count": len(anomalies),
            "overall_health": "healthy" if len(anomalies) == 0 else "anomalous"
        }
        
        return PatternAnalysis(
            spatial_patterns=spatial_patterns,
            temporal_patterns=temporal_patterns,
            correlations=correlations,
            anomalies=anomalies,
            summary=summary
        )
    
    def reset(self) -> None:
        """Reset pattern history."""
        self._history.clear()


# ============================================================================
# CONSCIOUSNESS TRACKER
# ============================================================================

@dataclass
class ConsciousnessState:
    """Current consciousness state."""
    awareness_level: float
    focus_intensity: float
    coherence_score: float
    entropy: float
    meta_awareness: Dict[str, Any]
    timestamp: datetime


class ConsciousnessTracker:
    """
    Tracks consciousness state over time.
    
    Monitors:
    - Awareness levels
    - Focus intensity
    - Coherence scores
    - Information entropy
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._history: List[ConsciousnessState] = []
    
    def _compute_awareness(
        self,
        consciousness_input: ConsciousnessInput
    ) -> float:
        """Compute awareness level from activation pattern."""
        activation = consciousness_input.activation_pattern
        
        # Awareness proportional to activation magnitude
        awareness = np.tanh(np.linalg.norm(activation) / 10.0)
        return float(awareness)
    
    def _compute_focus(
        self,
        consciousness_input: ConsciousnessInput
    ) -> float:
        """Compute focus intensity."""
        if consciousness_input.attention_focus is None:
            return 0.5
        
        focus = consciousness_input.attention_focus
        # Focus intensity is mean of focus vector
        intensity = np.mean(focus)
        return float(np.clip(intensity, 0, 1))
    
    def _compute_coherence(
        self,
        consciousness_input: ConsciousnessInput
    ) -> float:
        """Compute coherence score across modalities."""
        temporal = consciousness_input.temporal_context
        
        # Coherence based on temporal consistency
        if len(temporal) < 2:
            return 1.0
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(temporal) - 1):
            sim = np.dot(temporal[i], temporal[i + 1]) / (
                np.linalg.norm(temporal[i]) * np.linalg.norm(temporal[i + 1]) + 1e-10
            )
            similarities.append(sim)
        
        coherence = np.mean(similarities) if similarities else 1.0
        return float(np.clip(coherence, 0, 1))
    
    def _compute_entropy(
        self,
        consciousness_input: ConsciousnessInput
    ) -> float:
        """Compute information entropy."""
        activation = consciousness_input.activation_pattern
        
        # Normalize to probability distribution
        activation_abs = np.abs(activation) + 1e-10
        probs = activation_abs / np.sum(activation_abs)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log(probs))
        return float(entropy)
    
    def track(
        self,
        consciousness_input: ConsciousnessInput
    ) -> ConsciousnessState:
        """Track current consciousness state."""
        state = ConsciousnessState(
            awareness_level=self._compute_awareness(consciousness_input),
            focus_intensity=self._compute_focus(consciousness_input),
            coherence_score=self._compute_coherence(consciousness_input),
            entropy=self._compute_entropy(consciousness_input),
            meta_awareness={
                "temporal_depth": len(consciousness_input.temporal_context),
                "spatial_complexity": float(np.std(consciousness_input.spatial_features)),
                "source": consciousness_input.meta_state.get("source_metadata", {})
            },
            timestamp=datetime.utcnow()
        )
        
        # Update history
        self._history.append(state)
        if len(self._history) > self.history_size:
            self._history.pop(0)
        
        return state
    
    def get_trend(self, window: int = 10) -> Dict[str, str]:
        """Get trend of consciousness metrics."""
        if len(self._history) < 2:
            return {"overall": "insufficient_data"}
        
        recent = self._history[-window:]
        
        def trend_direction(values):
            if len(values) < 2:
                return "stable"
            diff = values[-1] - values[0]
            if diff > 0.1:
                return "increasing"
            elif diff < -0.1:
                return "decreasing"
            return "stable"
        
        return {
            "awareness": trend_direction([s.awareness_level for s in recent]),
            "focus": trend_direction([s.focus_intensity for s in recent]),
            "coherence": trend_direction([s.coherence_score for s in recent]),
            "entropy": trend_direction([s.entropy for s in recent])
        }
    
    def reset(self) -> None:
        """Reset tracking history."""
        self._history.clear()


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================

@dataclass
class Checkpoint:
    """System checkpoint."""
    version: int
    timestamp: datetime
    state_bridge_buffer: List[np.ndarray]
    pattern_history: List[ConsciousnessInput]
    consciousness_history: List[ConsciousnessState]
    orchestration_state: Dict[str, Any]
    metadata: Dict[str, Any]


class CheckpointManager:
    """
    Manages checkpoint saving and loading for system continuity.
    
    Provides:
    - Periodic checkpoint creation
    - State restoration
    - Checkpoint validation
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._version = 0
    
    def save(
        self,
        state_bridge: StateBridge,
        pattern_extractor: PatternExtractor,
        consciousness_tracker: ConsciousnessTracker,
        orchestration_state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save system checkpoint."""
        self._version += 1
        
        checkpoint = Checkpoint(
            version=self._version,
            timestamp=datetime.utcnow(),
            state_bridge_buffer=list(state_bridge._temporal_buffer),
            pattern_history=list(pattern_extractor._history),
            consciousness_history=list(consciousness_tracker._history),
            orchestration_state=orchestration_state,
            metadata=metadata or {}
        )
        
        filename = f"checkpoint_v{self._version}_{checkpoint.timestamp.strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = self.checkpoint_dir / filename
        
        with open(filepath, "wb") as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"Checkpoint saved: {filepath}")
        return str(filepath)
    
    def load(self, filepath: str) -> Checkpoint:
        """Load checkpoint from file."""
        with open(filepath, "rb") as f:
            checkpoint = pickle.load(f)
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint
    
    def restore(
        self,
        checkpoint: Checkpoint,
        state_bridge: StateBridge,
        pattern_extractor: PatternExtractor,
        consciousness_tracker: ConsciousnessTracker
    ) -> None:
        """Restore system state from checkpoint."""
        state_bridge._temporal_buffer = list(checkpoint.state_bridge_buffer)
        pattern_extractor._history = list(checkpoint.pattern_history)
        consciousness_tracker._history = list(checkpoint.consciousness_history)
        
        self._version = checkpoint.version
        logger.info(f"System restored to version {checkpoint.version}")
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints."""
        checkpoints = []
        for filepath in self.checkpoint_dir.glob("checkpoint_*.pkl"):
            try:
                with open(filepath, "rb") as f:
                    cp = pickle.load(f)
                checkpoints.append({
                    "filepath": str(filepath),
                    "version": cp.version,
                    "timestamp": cp.timestamp.isoformat(),
                    "metadata": cp.metadata
                })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {filepath}: {e}")
        
        return sorted(checkpoints, key=lambda x: x["version"], reverse=True)
    
    def get_latest(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]["filepath"]
        return None


# ============================================================================
# VICTOR SYSTEM
# ============================================================================

class VictorSystem:
    """
    Unified system combining all VICTOR components.
    
    Integrates:
    - State bridge for transformer → consciousness conversion
    - Pattern extractor for analysis
    - Consciousness tracker for state monitoring
    - Orchestration engine for infrastructure
    - Checkpoint manager for continuity
    """
    
    def __init__(
        self,
        node_id: str,
        config: Optional[DeploymentConfig] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.node_id = node_id
        self.config = config or DeploymentConfig.dev()
        
        # Core components
        self.state_bridge = StateBridge()
        self.pattern_extractor = PatternExtractor()
        self.consciousness_tracker = ConsciousnessTracker()
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # Infrastructure (initialized lazily)
        self.orchestration: Optional[OrchestrationEngine] = None
        
        # State
        self._initialized = False
        self._generation_count = 0
        self._last_checkpoint_generation = 0
    
    async def initialize(self) -> bool:
        """Initialize the Victor system."""
        logger.info(f"Initializing Victor system on node {self.node_id}")
        logger.info(f"Deployment config: {self.config.mode.value}")
        
        # Initialize orchestration engine
        self.orchestration = await create_orchestration_engine(self.node_id)
        
        # Try to restore from latest checkpoint
        latest = self.checkpoint_manager.get_latest()
        if latest:
            try:
                checkpoint = self.checkpoint_manager.load(latest)
                self.checkpoint_manager.restore(
                    checkpoint,
                    self.state_bridge,
                    self.pattern_extractor,
                    self.consciousness_tracker
                )
                logger.info("System restored from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to restore from checkpoint: {e}")
        
        self._initialized = True
        logger.info("Victor system initialized")
        return True
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the system."""
        logger.info("Shutting down Victor system")
        
        # Save final checkpoint
        if self.orchestration:
            health = await self.orchestration.get_system_health()
            self.checkpoint_manager.save(
                self.state_bridge,
                self.pattern_extractor,
                self.consciousness_tracker,
                health,
                {"shutdown_reason": "graceful"}
            )
        
        # Shutdown orchestration
        if self.orchestration:
            await self.orchestration.shutdown()
        
        self._initialized = False
        logger.info("Victor system shutdown complete")
    
    async def generate(
        self,
        transformer_output: TransformerOutput
    ) -> Dict[str, Any]:
        """
        Run full generation cycle with consciousness tracking.
        
        Steps:
        1. Convert transformer output to consciousness input
        2. Analyze patterns
        3. Track consciousness state
        4. Store results
        5. Checkpoint if needed
        """
        if not self._initialized:
            raise RuntimeError("System not initialized")
        
        self._generation_count += 1
        
        # Step 1: State bridge conversion
        consciousness_input = self.state_bridge.convert(transformer_output)
        
        # Step 2: Pattern analysis
        pattern_analysis = self.pattern_extractor.analyze(consciousness_input)
        
        # Step 3: Consciousness tracking
        consciousness_state = self.consciousness_tracker.track(consciousness_input)
        
        # Step 4: Store results
        if self.orchestration:
            generation_id = f"gen_{self._generation_count}"
            await self.orchestration.store_data(
                generation_id,
                {
                    "patterns": pattern_analysis.summary,
                    "consciousness": {
                        "awareness": consciousness_state.awareness_level,
                        "focus": consciousness_state.focus_intensity,
                        "coherence": consciousness_state.coherence_score,
                        "entropy": consciousness_state.entropy
                    }
                }
            )
            
            # Store to time-series for telemetry
            await self.orchestration.backends["timeseries"].write_point(
                "consciousness_metrics",
                {"node": self.node_id},
                {
                    "awareness": consciousness_state.awareness_level,
                    "focus": consciousness_state.focus_intensity,
                    "coherence": consciousness_state.coherence_score,
                    "entropy": consciousness_state.entropy
                }
            )
        
        # Step 5: Checkpoint if needed
        if self._should_checkpoint():
            await self._create_checkpoint()
        
        return {
            "generation_id": self._generation_count,
            "consciousness_state": {
                "awareness_level": consciousness_state.awareness_level,
                "focus_intensity": consciousness_state.focus_intensity,
                "coherence_score": consciousness_state.coherence_score,
                "entropy": consciousness_state.entropy
            },
            "pattern_analysis": {
                "spatial_pattern_count": len(pattern_analysis.spatial_patterns),
                "temporal_pattern_count": len(pattern_analysis.temporal_patterns),
                "anomaly_count": len(pattern_analysis.anomalies),
                "correlations": pattern_analysis.correlations
            },
            "trend": self.consciousness_tracker.get_trend(),
            "health": pattern_analysis.summary["overall_health"]
        }
    
    def _should_checkpoint(self) -> bool:
        """Check if checkpoint should be created."""
        generations_since = self._generation_count - self._last_checkpoint_generation
        return generations_since >= self.config.checkpoint_interval
    
    async def _create_checkpoint(self) -> None:
        """Create system checkpoint."""
        if self.orchestration:
            health = await self.orchestration.get_system_health()
            self.checkpoint_manager.save(
                self.state_bridge,
                self.pattern_extractor,
                self.consciousness_tracker,
                health,
                {
                    "generation_count": self._generation_count,
                    "deployment_mode": self.config.mode.value
                }
            )
            self._last_checkpoint_generation = self._generation_count
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "node_id": self.node_id,
            "initialized": self._initialized,
            "generation_count": self._generation_count,
            "deployment_mode": self.config.mode.value,
            "consciousness_trend": self.consciousness_tracker.get_trend(),
            "checkpoints": len(self.checkpoint_manager.list_checkpoints())
        }
        
        if self.orchestration:
            status["infrastructure"] = await self.orchestration.get_system_health()
        
        return status
    
    def reset(self) -> None:
        """Reset all components."""
        self.state_bridge.reset()
        self.pattern_extractor.reset()
        self.consciousness_tracker.reset()
        self._generation_count = 0
        self._last_checkpoint_generation = 0
        logger.info("Victor system reset")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point for running Victor system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="VICTOR Autonomous Intelligence System")
    parser.add_argument(
        "--mode",
        choices=["dev", "prod", "edge"],
        default="dev",
        help="Deployment mode"
    )
    parser.add_argument(
        "--node-id",
        default="victor-node-0",
        help="Node identifier"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpoints",
        help="Checkpoint directory"
    )
    
    args = parser.parse_args()
    
    # Select deployment config
    config_map = {
        "dev": DeploymentConfig.dev,
        "prod": DeploymentConfig.prod,
        "edge": DeploymentConfig.edge
    }
    config = config_map[args.mode]()
    
    # Create and initialize system
    system = VictorSystem(
        node_id=args.node_id,
        config=config,
        checkpoint_dir=args.checkpoint_dir
    )
    
    try:
        await system.initialize()
        
        # Demo: Run a few generations with synthetic data
        logger.info("Running demo generations...")
        
        for i in range(5):
            # Create synthetic transformer output
            embeddings = np.random.randn(16, 768).astype(np.float32)
            attention = np.random.rand(8, 16, 16).astype(np.float32)
            attention = attention / attention.sum(axis=-1, keepdims=True)
            
            output = TransformerOutput(
                embeddings=embeddings,
                attention_weights=attention,
                metadata={"iteration": i}
            )
            
            result = await system.generate(output)
            logger.info(
                f"Generation {result['generation_id']}: "
                f"awareness={result['consciousness_state']['awareness_level']:.3f}, "
                f"coherence={result['consciousness_state']['coherence_score']:.3f}"
            )
        
        # Print final status
        status = await system.get_system_status()
        logger.info(f"System status: {json.dumps(status, indent=2, default=str)}")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
