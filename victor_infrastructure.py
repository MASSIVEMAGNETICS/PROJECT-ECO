"""
VICTOR Infrastructure Layer
Enterprise-grade autonomous intelligence system infrastructure.

This module provides:
- Circuit breakers for failure isolation
- Distributed state management with versioning
- Multiple persistence backends (Vector, Graph, Time-Series)
- Health monitoring and task execution with retries
- Orchestration engine coordinating all subsystems
"""

import asyncio
import time
import logging
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("victor.infrastructure")


# ============================================================================
# CIRCUIT BREAKER PATTERN
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failure mode, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker implementation for failure isolation.
    
    Prevents cascade failures by tracking errors and temporarily
    blocking requests to failing services.
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
        
    async def can_execute(self) -> bool:
        """Check if execution is allowed based on circuit state."""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info(f"Circuit {self.name} transitioning to HALF_OPEN")
                    return True
                return False
            else:  # HALF_OPEN
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
    
    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    async def record_success(self) -> None:
        """Record successful execution."""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.half_open_max_calls:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logger.info(f"Circuit {self.name} recovered to CLOSED")
            elif self.state == CircuitState.CLOSED:
                self.failure_count = max(0, self.failure_count - 1)
    
    async def record_failure(self) -> None:
        """Record failed execution."""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                self.success_count = 0
                logger.warning(f"Circuit {self.name} reopened due to failure")
            elif self.state == CircuitState.CLOSED:
                if self.failure_count >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
                    logger.warning(f"Circuit {self.name} opened after {self.failure_count} failures")
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if not await self.can_execute():
            raise CircuitOpenError(f"Circuit {self.name} is open")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure()
            raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


# ============================================================================
# DISTRIBUTED STATE MANAGEMENT
# ============================================================================

@dataclass
class StateVersion:
    """State version with metadata."""
    version: int
    timestamp: datetime
    checksum: str
    data: Dict[str, Any]


class DistributedStateManager:
    """
    Manages distributed state with versioning and consistency.
    
    Features:
    - Version tracking for all state changes
    - Checksum validation for data integrity
    - In-memory caching with fallback capability
    - State replication support
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.current_version = 0
        self.state_history: List[StateVersion] = []
        self.current_state: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._subscribers: List[Callable] = []
        
    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 checksum of state data."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    async def update_state(self, updates: Dict[str, Any]) -> StateVersion:
        """Update state with versioning."""
        async with self._lock:
            self.current_version += 1
            self.current_state.update(updates)
            
            version = StateVersion(
                version=self.current_version,
                timestamp=datetime.utcnow(),
                checksum=self._compute_checksum(self.current_state),
                data=dict(self.current_state)
            )
            
            self.state_history.append(version)
            
            # Notify subscribers
            for subscriber in self._subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(version)
                    else:
                        subscriber(version)
                except Exception as e:
                    logger.error(f"Error notifying subscriber: {e}")
            
            logger.debug(f"State updated to version {self.current_version}")
            return version
    
    async def get_state(self, key: Optional[str] = None) -> Any:
        """Get current state or specific key."""
        async with self._lock:
            if key is None:
                return dict(self.current_state)
            return self.current_state.get(key)
    
    async def get_version(self, version: int) -> Optional[StateVersion]:
        """Get specific version of state."""
        async with self._lock:
            for sv in self.state_history:
                if sv.version == version:
                    return sv
            return None
    
    async def rollback(self, version: int) -> bool:
        """Rollback to a specific version."""
        async with self._lock:
            target = None
            for sv in self.state_history:
                if sv.version == version:
                    target = sv
                    break
            
            if target is None:
                return False
            
            self.current_state = dict(target.data)
            self.current_version = version
            logger.info(f"State rolled back to version {version}")
            return True
    
    def subscribe(self, callback: Callable) -> None:
        """Subscribe to state changes."""
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable) -> None:
        """Unsubscribe from state changes."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)


# ============================================================================
# PERSISTENCE BACKENDS
# ============================================================================

class PersistenceBackend(ABC):
    """Abstract base class for persistence backends."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to backend."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from backend."""
        pass
    
    @abstractmethod
    async def store(self, key: str, value: Any) -> bool:
        """Store data."""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete data."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check backend health."""
        pass


class InMemoryBackend(PersistenceBackend):
    """In-memory persistence backend for development and fallback."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._connected = False
    
    async def connect(self) -> bool:
        self._connected = True
        logger.info("InMemory backend connected")
        return True
    
    async def disconnect(self) -> None:
        self._connected = False
        logger.info("InMemory backend disconnected")
    
    async def store(self, key: str, value: Any) -> bool:
        if not self._connected:
            return False
        self._data[key] = value
        return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        if not self._connected:
            return None
        return self._data.get(key)
    
    async def delete(self, key: str) -> bool:
        if not self._connected:
            return False
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    async def health_check(self) -> bool:
        return self._connected


class VectorDBBackend(PersistenceBackend):
    """
    Vector database backend for embeddings storage.
    
    Supports Pinecone/Weaviate-style operations.
    In production, replace with actual client.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self._vectors: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._connected = False
    
    async def connect(self) -> bool:
        # In production, connect to actual vector DB
        self._connected = True
        logger.info(f"VectorDB backend connected to {self.host}:{self.port}")
        return True
    
    async def disconnect(self) -> None:
        self._connected = False
        logger.info("VectorDB backend disconnected")
    
    async def store(self, key: str, value: Any) -> bool:
        if not self._connected:
            return False
        if isinstance(value, dict):
            self._vectors[key] = value.get("vector", [])
            self._metadata[key] = value.get("metadata", {})
        else:
            self._vectors[key] = value if isinstance(value, list) else []
        return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        if not self._connected:
            return None
        if key in self._vectors:
            return {
                "vector": self._vectors[key],
                "metadata": self._metadata.get(key, {})
            }
        return None
    
    async def delete(self, key: str) -> bool:
        if not self._connected:
            return False
        deleted = False
        if key in self._vectors:
            del self._vectors[key]
            deleted = True
        if key in self._metadata:
            del self._metadata[key]
        return deleted
    
    async def health_check(self) -> bool:
        return self._connected
    
    async def similarity_search(
        self, 
        query_vector: List[float], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar vectors using cosine similarity."""
        if not self._connected or not self._vectors:
            return []
        
        results = []
        query_norm = sum(x * x for x in query_vector) ** 0.5
        
        for key, vec in self._vectors.items():
            if len(vec) != len(query_vector) or query_norm == 0:
                continue
            
            vec_norm = sum(x * x for x in vec) ** 0.5
            if vec_norm == 0:
                continue
            
            dot_product = sum(q * v for q, v in zip(query_vector, vec))
            similarity = dot_product / (query_norm * vec_norm)
            
            results.append({
                "id": key,
                "score": similarity,
                "metadata": self._metadata.get(key, {})
            })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]


class GraphDBBackend(PersistenceBackend):
    """
    Graph database backend for relationship storage.
    
    Supports Neo4j-style operations.
    In production, replace with actual client.
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687"):
        self.uri = uri
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._edges: List[Dict[str, Any]] = []
        self._connected = False
    
    async def connect(self) -> bool:
        self._connected = True
        logger.info(f"GraphDB backend connected to {self.uri}")
        return True
    
    async def disconnect(self) -> None:
        self._connected = False
        logger.info("GraphDB backend disconnected")
    
    async def store(self, key: str, value: Any) -> bool:
        if not self._connected:
            return False
        self._nodes[key] = value if isinstance(value, dict) else {"value": value}
        return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        if not self._connected:
            return None
        return self._nodes.get(key)
    
    async def delete(self, key: str) -> bool:
        if not self._connected:
            return False
        if key in self._nodes:
            del self._nodes[key]
            self._edges = [e for e in self._edges 
                         if e["source"] != key and e["target"] != key]
            return True
        return False
    
    async def health_check(self) -> bool:
        return self._connected
    
    async def create_relationship(
        self, 
        source: str, 
        target: str, 
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between nodes."""
        if not self._connected:
            return False
        
        if source not in self._nodes or target not in self._nodes:
            return False
        
        self._edges.append({
            "source": source,
            "target": target,
            "type": relationship_type,
            "properties": properties or {}
        })
        return True
    
    async def get_relationships(
        self, 
        node_id: str, 
        direction: str = "both"
    ) -> List[Dict[str, Any]]:
        """Get relationships for a node."""
        if not self._connected:
            return []
        
        results = []
        for edge in self._edges:
            if direction in ("both", "outgoing") and edge["source"] == node_id:
                results.append(edge)
            elif direction in ("both", "incoming") and edge["target"] == node_id:
                results.append(edge)
        return results


class TimeSeriesDBBackend(PersistenceBackend):
    """
    Time-series database backend for telemetry.
    
    Supports InfluxDB-style operations.
    In production, replace with actual client.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8086):
        self.host = host
        self.port = port
        self._measurements: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._connected = False
    
    async def connect(self) -> bool:
        self._connected = True
        logger.info(f"TimeSeriesDB backend connected to {self.host}:{self.port}")
        return True
    
    async def disconnect(self) -> None:
        self._connected = False
        logger.info("TimeSeriesDB backend disconnected")
    
    async def store(self, key: str, value: Any) -> bool:
        if not self._connected:
            return False
        
        point = {
            "timestamp": datetime.utcnow().isoformat(),
            "value": value
        }
        self._measurements[key].append(point)
        return True
    
    async def retrieve(self, key: str) -> Optional[Any]:
        if not self._connected:
            return None
        return self._measurements.get(key, [])
    
    async def delete(self, key: str) -> bool:
        if not self._connected:
            return False
        if key in self._measurements:
            del self._measurements[key]
            return True
        return False
    
    async def health_check(self) -> bool:
        return self._connected
    
    async def write_point(
        self,
        measurement: str,
        tags: Dict[str, str],
        fields: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ) -> bool:
        """Write a data point."""
        if not self._connected:
            return False
        
        point = {
            "timestamp": (timestamp or datetime.utcnow()).isoformat(),
            "tags": tags,
            "fields": fields
        }
        self._measurements[measurement].append(point)
        return True
    
    async def query(
        self,
        measurement: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Query data points."""
        if not self._connected:
            return []
        
        results = []
        for point in self._measurements.get(measurement, []):
            # Time filter
            point_time = datetime.fromisoformat(point["timestamp"])
            if start_time and point_time < start_time:
                continue
            if end_time and point_time > end_time:
                continue
            
            # Tag filter
            if tags:
                point_tags = point.get("tags", {})
                if not all(point_tags.get(k) == v for k, v in tags.items()):
                    continue
            
            results.append(point)
        
        return results


# ============================================================================
# HEALTH MONITORING
# ============================================================================

@dataclass
class HealthStatus:
    """Health status of a component."""
    component: str
    healthy: bool
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class HealthMonitor:
    """
    Monitors health of all system components.
    
    Performs periodic health checks and maintains status history.
    """
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._components: Dict[str, Callable] = {}
        self._status: Dict[str, HealthStatus] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    def register_component(
        self, 
        name: str, 
        health_check: Callable[[], bool]
    ) -> None:
        """Register a component for health monitoring."""
        self._components[name] = health_check
        logger.info(f"Registered component for health monitoring: {name}")
    
    def unregister_component(self, name: str) -> None:
        """Unregister a component from health monitoring."""
        if name in self._components:
            del self._components[name]
            logger.info(f"Unregistered component: {name}")
    
    async def check_component(self, name: str) -> HealthStatus:
        """Check health of a specific component."""
        if name not in self._components:
            return HealthStatus(
                component=name,
                healthy=False,
                last_check=datetime.utcnow(),
                error="Component not registered"
            )
        
        try:
            check_func = self._components[name]
            if asyncio.iscoroutinefunction(check_func):
                healthy = await check_func()
            else:
                healthy = check_func()
            
            status = HealthStatus(
                component=name,
                healthy=healthy,
                last_check=datetime.utcnow()
            )
        except Exception as e:
            status = HealthStatus(
                component=name,
                healthy=False,
                last_check=datetime.utcnow(),
                error=str(e)
            )
        
        self._status[name] = status
        return status
    
    async def check_all(self) -> Dict[str, HealthStatus]:
        """Check health of all registered components."""
        for name in self._components:
            await self.check_component(name)
        return dict(self._status)
    
    def get_status(self, name: Optional[str] = None) -> Any:
        """Get current health status."""
        if name:
            return self._status.get(name)
        return dict(self._status)
    
    async def _monitoring_loop(self) -> None:
        """Internal monitoring loop."""
        while self._running:
            await self.check_all()
            await asyncio.sleep(self.check_interval)
    
    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")


# ============================================================================
# TASK EXECUTION WITH RETRIES
# ============================================================================

@dataclass
class TaskConfig:
    """Configuration for task execution."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class TaskExecutor:
    """
    Executes tasks with retry logic and exponential backoff.
    
    Supports async and sync functions with configurable retry behavior.
    """
    
    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self._tasks: Dict[str, Dict[str, Any]] = {}
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff."""
        delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        return min(delay, self.config.max_delay)
    
    async def execute(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a task with retries."""
        self._tasks[task_id] = {
            "status": "running",
            "attempts": 0,
            "started_at": datetime.utcnow()
        }
        
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            self._tasks[task_id]["attempts"] = attempt + 1
            
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                self._tasks[task_id]["status"] = "completed"
                self._tasks[task_id]["completed_at"] = datetime.utcnow()
                logger.info(f"Task {task_id} completed after {attempt + 1} attempts")
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Task {task_id} attempt {attempt + 1} failed: {e}"
                )
                
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.info(f"Retrying task {task_id} in {delay:.2f} seconds")
                    await asyncio.sleep(delay)
        
        self._tasks[task_id]["status"] = "failed"
        self._tasks[task_id]["error"] = str(last_error)
        logger.error(f"Task {task_id} failed after {self.config.max_retries + 1} attempts")
        raise last_error if last_error else RuntimeError("Task failed")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a task."""
        return self._tasks.get(task_id)


# ============================================================================
# ORCHESTRATION ENGINE
# ============================================================================

class OrchestrationEngine:
    """
    Coordinates all infrastructure subsystems.
    
    Manages:
    - Circuit breakers for each service
    - Distributed state
    - Multiple persistence backends
    - Health monitoring
    - Task execution
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        
        # Initialize components
        self.state_manager = DistributedStateManager(node_id)
        self.health_monitor = HealthMonitor()
        self.task_executor = TaskExecutor()
        
        # Circuit breakers for each backend
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Persistence backends
        self.backends: Dict[str, PersistenceBackend] = {
            "memory": InMemoryBackend(),
            "vector": VectorDBBackend(),
            "graph": GraphDBBackend(),
            "timeseries": TimeSeriesDBBackend()
        }
        
        # Initialize circuit breakers
        for name in self.backends:
            self.circuit_breakers[name] = CircuitBreaker(
                f"backend_{name}",
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30.0)
            )
        
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all backends and start monitoring."""
        logger.info(f"Initializing orchestration engine on node {self.node_id}")
        
        # Connect to all backends
        for name, backend in self.backends.items():
            try:
                cb = self.circuit_breakers[name]
                await cb.execute(backend.connect)
                logger.info(f"Backend {name} connected")
            except Exception as e:
                logger.warning(f"Failed to connect backend {name}: {e}")
        
        # Register health checks
        for name, backend in self.backends.items():
            self.health_monitor.register_component(
                f"backend_{name}",
                backend.health_check
            )
        
        # Start health monitoring
        await self.health_monitor.start()
        
        self._initialized = True
        logger.info("Orchestration engine initialized")
        return True
    
    async def shutdown(self) -> None:
        """Gracefully shutdown all components."""
        logger.info("Shutting down orchestration engine")
        
        # Stop health monitoring
        await self.health_monitor.stop()
        
        # Disconnect all backends
        for name, backend in self.backends.items():
            try:
                await backend.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting backend {name}: {e}")
        
        self._initialized = False
        logger.info("Orchestration engine shutdown complete")
    
    async def store_data(
        self,
        key: str,
        value: Any,
        backend: str = "memory"
    ) -> bool:
        """Store data with circuit breaker protection."""
        if backend not in self.backends:
            raise ValueError(f"Unknown backend: {backend}")
        
        cb = self.circuit_breakers[backend]
        storage = self.backends[backend]
        
        try:
            return await cb.execute(storage.store, key, value)
        except CircuitOpenError:
            # Fallback to memory backend
            if backend != "memory":
                logger.warning(f"Circuit open for {backend}, falling back to memory")
                return await self.backends["memory"].store(key, value)
            raise
    
    async def retrieve_data(
        self,
        key: str,
        backend: str = "memory"
    ) -> Optional[Any]:
        """Retrieve data with circuit breaker protection."""
        if backend not in self.backends:
            raise ValueError(f"Unknown backend: {backend}")
        
        cb = self.circuit_breakers[backend]
        storage = self.backends[backend]
        
        try:
            return await cb.execute(storage.retrieve, key)
        except CircuitOpenError:
            # Fallback to memory backend
            if backend != "memory":
                logger.warning(f"Circuit open for {backend}, falling back to memory")
                return await self.backends["memory"].retrieve(key)
            raise
    
    async def execute_task(
        self,
        task_id: str,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a task with retries."""
        return await self.task_executor.execute(task_id, func, *args, **kwargs)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        health = await self.health_monitor.check_all()
        
        circuit_status = {}
        for name, cb in self.circuit_breakers.items():
            circuit_status[name] = {
                "state": cb.state.value,
                "failure_count": cb.failure_count
            }
        
        return {
            "node_id": self.node_id,
            "initialized": self._initialized,
            "backends": {
                name: {
                    "healthy": status.healthy,
                    "last_check": status.last_check.isoformat() if status.last_check else None,
                    "error": status.error
                }
                for name, status in health.items()
            },
            "circuit_breakers": circuit_status,
            "state_version": self.state_manager.current_version
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

async def create_orchestration_engine(node_id: str) -> OrchestrationEngine:
    """Factory function to create and initialize orchestration engine."""
    engine = OrchestrationEngine(node_id)
    await engine.initialize()
    return engine


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    
    # State Management
    "DistributedStateManager",
    "StateVersion",
    
    # Persistence Backends
    "PersistenceBackend",
    "InMemoryBackend",
    "VectorDBBackend",
    "GraphDBBackend",
    "TimeSeriesDBBackend",
    
    # Health Monitoring
    "HealthMonitor",
    "HealthStatus",
    
    # Task Execution
    "TaskExecutor",
    "TaskConfig",
    
    # Orchestration
    "OrchestrationEngine",
    "create_orchestration_engine",
]
