"""
Singleton Pattern Implementations for Production Gate Node.

This module provides thread-safe singleton instances for expensive resources:
- ONNX sessions (detector, recognizer)
- Pre-allocated buffers
- Shared state managers

Design Principles:
1. Lazy initialization (don't load until needed)
2. Thread-safe (multiple threads can request instances)
3. Reusable (same instance across the application)
4. Cleanup support (proper resource release)
"""

import threading
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.
    
    Usage:
        class MySingleton(metaclass=SingletonMeta):
            pass
    """
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
    
    @classmethod
    def clear_instance(mcs, cls):
        """Clear a singleton instance (for testing/cleanup)."""
        with mcs._lock:
            if cls in mcs._instances:
                del mcs._instances[cls]


class ONNXSessionManager(metaclass=SingletonMeta):
    """
    Centralized manager for ONNX Runtime sessions.
    
    Benefits:
    - Single session per model (memory efficient)
    - Thread-safe session access
    - Lazy loading (models loaded on first use)
    - Provider optimization (GPU if available)
    
    Usage:
        manager = ONNXSessionManager()
        detector_session = manager.get_session('detector', 'path/to/model.onnx')
        recognizer_session = manager.get_session('recognizer', 'path/to/arcface.onnx')
    """
    
    def __init__(self):
        self._sessions: Dict[str, Any] = {}  # ort.InferenceSession stored here
        self._session_lock = threading.Lock()
        self._providers = self._get_optimal_providers()
        logger.info(f"ONNXSessionManager initialized with providers: {self._providers}")
    
    def _get_optimal_providers(self) -> list:
        """Get optimal execution providers based on hardware."""
        if ort is None:
            return []
        
        available = ort.get_available_providers()
        
        # Priority: CUDA > DirectML > CPU
        providers = []
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        return providers
    
    def get_session(self, name: str, model_path: str) -> Optional[Any]:
        """
        Get or create ONNX session for a model.
        
        Args:
            name: Unique identifier for this session (e.g., 'detector', 'recognizer')
            model_path: Path to the ONNX model file
        
        Returns:
            ONNX InferenceSession or None if failed
        """
        with self._session_lock:
            # Return existing session
            if name in self._sessions:
                return self._sessions[name]
            
            # Create new session
            if ort is None:
                logger.error("ONNX Runtime not available")
                return None
            
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return None
            
            try:
                # Session options for optimization
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = 4  # Parallel execution within ops
                sess_options.inter_op_num_threads = 2  # Parallel execution between ops
                
                # Enable memory optimizations
                sess_options.enable_mem_pattern = True
                sess_options.enable_cpu_mem_arena = True
                
                session = ort.InferenceSession(
                    model_path,
                    sess_options=sess_options,
                    providers=self._providers
                )
                
                self._sessions[name] = session
                logger.info(f"ONNX session '{name}' loaded from {model_path}")
                logger.info(f"  Using providers: {session.get_providers()}")
                
                return session
                
            except Exception as e:
                logger.error(f"Failed to create ONNX session '{name}': {e}")
                return None
    
    def get_input_name(self, name: str) -> Optional[str]:
        """Get input tensor name for a session."""
        session = self._sessions.get(name)
        if session:
            return session.get_inputs()[0].name
        return None
    
    def get_output_names(self, name: str) -> Optional[list]:
        """Get output tensor names for a session."""
        session = self._sessions.get(name)
        if session:
            return [o.name for o in session.get_outputs()]
        return None
    
    def cleanup(self):
        """Release all sessions."""
        with self._session_lock:
            self._sessions.clear()
            logger.info("ONNXSessionManager cleaned up")


class BufferPool(metaclass=SingletonMeta):
    """
    Pre-allocated buffer pool for zero-copy operations.
    
    Reduces GC pressure by reusing numpy arrays.
    
    Usage:
        pool = BufferPool()
        frame_buffer = pool.get_buffer('frame', (720, 1280, 3), np.uint8)
        # ... use buffer ...
        pool.release_buffer('frame')
    """
    
    def __init__(self):
        self._buffers: Dict[str, np.ndarray] = {}
        self._buffer_lock = threading.Lock()
        self._in_use: Dict[str, bool] = {}
    
    def get_buffer(
        self,
        name: str,
        shape: tuple,
        dtype: np.dtype = np.uint8
    ) -> np.ndarray:
        """
        Get or create a pre-allocated buffer.
        
        If buffer exists with correct shape/dtype, reuse it.
        Otherwise, create new buffer.
        """
        with self._buffer_lock:
            # Check if buffer exists and has correct spec
            if name in self._buffers:
                buf = self._buffers[name]
                if buf.shape == shape and buf.dtype == dtype:
                    self._in_use[name] = True
                    buf.fill(0)  # Clear buffer
                    return buf
            
            # Create new buffer
            buffer = np.zeros(shape, dtype=dtype)
            self._buffers[name] = buffer
            self._in_use[name] = True
            return buffer
    
    def release_buffer(self, name: str):
        """Mark buffer as available for reuse."""
        with self._buffer_lock:
            if name in self._in_use:
                self._in_use[name] = False
    
    def cleanup(self):
        """Release all buffers."""
        with self._buffer_lock:
            self._buffers.clear()
            self._in_use.clear()


class FrameCounter(metaclass=SingletonMeta):
    """
    Global frame counter for synchronization and debugging.
    
    Thread-safe counter that can be used across components.
    """
    
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()
        self._start_time = None
    
    def increment(self) -> int:
        """Increment and return new count."""
        with self._lock:
            if self._start_time is None:
                import time
                self._start_time = time.time()
            self._count += 1
            return self._count
    
    def get_count(self) -> int:
        """Get current count."""
        with self._lock:
            return self._count
    
    def get_fps(self) -> float:
        """Calculate FPS since start."""
        with self._lock:
            if self._start_time is None or self._count == 0:
                return 0.0
            import time
            elapsed = time.time() - self._start_time
            return self._count / elapsed if elapsed > 0 else 0.0
    
    def reset(self):
        """Reset counter."""
        with self._lock:
            self._count = 0
            self._start_time = None


# Convenience functions for global access
def get_onnx_manager() -> ONNXSessionManager:
    """Get the global ONNX session manager."""
    return ONNXSessionManager()


def get_buffer_pool() -> BufferPool:
    """Get the global buffer pool."""
    return BufferPool()


def get_frame_counter() -> FrameCounter:
    """Get the global frame counter."""
    return FrameCounter()


def cleanup_all():
    """Cleanup all singleton resources."""
    ONNXSessionManager().cleanup()
    BufferPool().cleanup()
    FrameCounter().reset()
    logger.info("All singleton resources cleaned up")
