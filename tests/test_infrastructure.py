"""
VICTOR Unit Tests
Tests for infrastructure and integration components.
"""

import asyncio
import numpy as np
import pytest

from victor_infrastructure import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    CircuitState,
    DistributedStateManager,
    HealthMonitor,
    InMemoryBackend,
    OrchestrationEngine,
    TaskConfig,
    TaskExecutor,
    VectorDBBackend,
)


# ============================================================================
# CIRCUIT BREAKER TESTS
# ============================================================================

class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""
    
    @pytest.fixture
    def circuit_breaker(self):
        """Create a circuit breaker with low thresholds for testing."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=0.1,
            half_open_max_calls=2
        )
        return CircuitBreaker("test-circuit", config)
    
    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, circuit_breaker):
        """Circuit breaker should start in closed state."""
        assert circuit_breaker.state == CircuitState.CLOSED
        assert await circuit_breaker.can_execute() is True
    
    @pytest.mark.asyncio
    async def test_opens_after_failures(self, circuit_breaker):
        """Circuit breaker should open after threshold failures."""
        # Record failures
        await circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitState.CLOSED
        
        await circuit_breaker.record_failure()
        assert circuit_breaker.state == CircuitState.OPEN
        assert await circuit_breaker.can_execute() is False
    
    @pytest.mark.asyncio
    async def test_recovers_after_timeout(self, circuit_breaker):
        """Circuit breaker should transition to half-open after timeout."""
        # Open the circuit
        for _ in range(2):
            await circuit_breaker.record_failure()
        
        assert circuit_breaker.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(0.15)
        
        # Should allow test call
        assert await circuit_breaker.can_execute() is True
        assert circuit_breaker.state == CircuitState.HALF_OPEN
    
    @pytest.mark.asyncio
    async def test_closes_after_successful_half_open(self, circuit_breaker):
        """Circuit should close after successful calls in half-open state."""
        # Open and wait for half-open
        for _ in range(2):
            await circuit_breaker.record_failure()
        await asyncio.sleep(0.15)
        await circuit_breaker.can_execute()
        
        # Record successful calls
        await circuit_breaker.record_success()
        await circuit_breaker.record_success()
        
        assert circuit_breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_execute_with_success(self, circuit_breaker):
        """Execute should run function and record success."""
        async def success_func():
            return "success"
        
        result = await circuit_breaker.execute(success_func)
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_execute_with_failure(self, circuit_breaker):
        """Execute should record failure on exception."""
        async def failing_func():
            raise ValueError("test error")
        
        with pytest.raises(ValueError):
            await circuit_breaker.execute(failing_func)
        
        assert circuit_breaker.failure_count == 1


# ============================================================================
# STATE MANAGER TESTS
# ============================================================================

class TestDistributedStateManager:
    """Tests for DistributedStateManager class."""
    
    @pytest.fixture
    def state_manager(self):
        """Create a state manager for testing."""
        return DistributedStateManager("test-node")
    
    @pytest.mark.asyncio
    async def test_update_state(self, state_manager):
        """Should update state and increment version."""
        version = await state_manager.update_state({"key": "value"})
        
        assert version.version == 1
        assert version.data["key"] == "value"
        assert state_manager.current_version == 1
    
    @pytest.mark.asyncio
    async def test_get_state(self, state_manager):
        """Should retrieve current state."""
        await state_manager.update_state({"key1": "value1"})
        await state_manager.update_state({"key2": "value2"})
        
        state = await state_manager.get_state()
        assert state["key1"] == "value1"
        assert state["key2"] == "value2"
        
        value = await state_manager.get_state("key1")
        assert value == "value1"
    
    @pytest.mark.asyncio
    async def test_rollback(self, state_manager):
        """Should rollback to previous version."""
        await state_manager.update_state({"key": "original"})
        await state_manager.update_state({"key": "modified"})
        
        success = await state_manager.rollback(1)
        assert success is True
        
        state = await state_manager.get_state()
        assert state["key"] == "original"
    
    @pytest.mark.asyncio
    async def test_checksum_consistency(self, state_manager):
        """Checksum should be consistent for same data."""
        version1 = await state_manager.update_state({"key": "value"})
        
        # Reset and recreate same state
        state_manager2 = DistributedStateManager("test-node-2")
        version2 = await state_manager2.update_state({"key": "value"})
        
        assert version1.checksum == version2.checksum


# ============================================================================
# PERSISTENCE BACKEND TESTS
# ============================================================================

class TestInMemoryBackend:
    """Tests for InMemoryBackend class."""
    
    @pytest.fixture
    def backend(self):
        """Create and connect backend."""
        return InMemoryBackend()
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, backend):
        """Should connect and disconnect properly."""
        assert await backend.connect() is True
        assert await backend.health_check() is True
        
        await backend.disconnect()
        assert await backend.health_check() is False
    
    @pytest.mark.asyncio
    async def test_store_retrieve(self, backend):
        """Should store and retrieve data."""
        await backend.connect()
        
        await backend.store("key1", {"data": "value"})
        result = await backend.retrieve("key1")
        
        assert result == {"data": "value"}
    
    @pytest.mark.asyncio
    async def test_delete(self, backend):
        """Should delete data."""
        await backend.connect()
        
        await backend.store("key1", "value")
        result = await backend.delete("key1")
        assert result is True
        
        result = await backend.retrieve("key1")
        assert result is None


class TestVectorDBBackend:
    """Tests for VectorDBBackend class."""
    
    @pytest.fixture
    def backend(self):
        """Create and connect backend."""
        return VectorDBBackend()
    
    @pytest.mark.asyncio
    async def test_store_vector(self, backend):
        """Should store vector with metadata."""
        await backend.connect()
        
        vector_data = {
            "vector": [0.1, 0.2, 0.3],
            "metadata": {"label": "test"}
        }
        await backend.store("vec1", vector_data)
        
        result = await backend.retrieve("vec1")
        assert result["vector"] == [0.1, 0.2, 0.3]
        assert result["metadata"]["label"] == "test"
    
    @pytest.mark.asyncio
    async def test_similarity_search(self, backend):
        """Should perform similarity search."""
        await backend.connect()
        
        # Store some vectors
        await backend.store("vec1", {"vector": [1.0, 0.0, 0.0], "metadata": {"id": 1}})
        await backend.store("vec2", {"vector": [0.9, 0.1, 0.0], "metadata": {"id": 2}})
        await backend.store("vec3", {"vector": [0.0, 1.0, 0.0], "metadata": {"id": 3}})
        
        # Search for similar vectors
        results = await backend.similarity_search([1.0, 0.0, 0.0], top_k=2)
        
        assert len(results) == 2
        assert results[0]["id"] == "vec1"  # Most similar


# ============================================================================
# TASK EXECUTOR TESTS
# ============================================================================

class TestTaskExecutor:
    """Tests for TaskExecutor class."""
    
    @pytest.fixture
    def executor(self):
        """Create executor with low retry counts."""
        config = TaskConfig(
            max_retries=2,
            base_delay=0.01,
            max_delay=0.1
        )
        return TaskExecutor(config)
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, executor):
        """Should execute and return result."""
        async def task():
            return "result"
        
        result = await executor.execute("task1", task)
        assert result == "result"
        
        status = executor.get_task_status("task1")
        assert status["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, executor):
        """Should retry on failure."""
        attempt_count = 0
        
        async def flaky_task():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("temporary failure")
            return "success"
        
        result = await executor.execute("task2", flaky_task)
        assert result == "success"
        assert attempt_count == 2
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, executor):
        """Should fail after max retries."""
        async def always_fails():
            raise ValueError("permanent failure")
        
        with pytest.raises(ValueError):
            await executor.execute("task3", always_fails)
        
        status = executor.get_task_status("task3")
        assert status["status"] == "failed"
        assert status["attempts"] == 3  # 1 initial + 2 retries


# ============================================================================
# HEALTH MONITOR TESTS
# ============================================================================

class TestHealthMonitor:
    """Tests for HealthMonitor class."""
    
    @pytest.fixture
    def monitor(self):
        """Create health monitor."""
        return HealthMonitor(check_interval=0.1)
    
    @pytest.mark.asyncio
    async def test_register_component(self, monitor):
        """Should register and check components."""
        monitor.register_component("test-component", lambda: True)
        
        status = await monitor.check_component("test-component")
        assert status.healthy is True
    
    @pytest.mark.asyncio
    async def test_unhealthy_component(self, monitor):
        """Should detect unhealthy components."""
        monitor.register_component("failing-component", lambda: False)
        
        status = await monitor.check_component("failing-component")
        assert status.healthy is False
    
    @pytest.mark.asyncio
    async def test_check_all(self, monitor):
        """Should check all registered components."""
        monitor.register_component("comp1", lambda: True)
        monitor.register_component("comp2", lambda: True)
        monitor.register_component("comp3", lambda: False)
        
        all_status = await monitor.check_all()
        
        assert all_status["comp1"].healthy is True
        assert all_status["comp2"].healthy is True
        assert all_status["comp3"].healthy is False


# ============================================================================
# ORCHESTRATION ENGINE TESTS
# ============================================================================

class TestOrchestrationEngine:
    """Tests for OrchestrationEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create orchestration engine."""
        return OrchestrationEngine("test-node")
    
    @pytest.mark.asyncio
    async def test_initialize_shutdown(self, engine):
        """Should initialize and shutdown cleanly."""
        success = await engine.initialize()
        assert success is True
        assert engine._initialized is True
        
        await engine.shutdown()
        assert engine._initialized is False
    
    @pytest.mark.asyncio
    async def test_store_retrieve_data(self, engine):
        """Should store and retrieve data."""
        await engine.initialize()
        
        await engine.store_data("key1", {"value": "test"})
        result = await engine.retrieve_data("key1")
        
        assert result == {"value": "test"}
        
        await engine.shutdown()
    
    @pytest.mark.asyncio
    async def test_system_health(self, engine):
        """Should return comprehensive health status."""
        await engine.initialize()
        
        health = await engine.get_system_health()
        
        assert health["node_id"] == "test-node"
        assert health["initialized"] is True
        assert "backends" in health
        assert "circuit_breakers" in health
        
        await engine.shutdown()
