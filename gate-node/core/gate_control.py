"""
Gate Controller and Decision Engine.
Handles gate hardware control and access decisions.
"""

import threading
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Callable
import logging


logger = logging.getLogger(__name__)


class GateDecision(Enum):
    """Decision result for gate access."""
    AUTHORIZED = "AUTHORIZED"
    UNKNOWN = "UNKNOWN"
    WANTED = "WANTED"


class GateState(Enum):
    """Gate physical state."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    ERROR = "ERROR"


class GateAction(Enum):
    """Gate action to take."""
    OPEN = "OPEN"
    CLOSE = "CLOSE"  # Keep closed / deny


@dataclass
class AccessDecision:
    """Result of access decision."""
    action: GateAction
    decision: GateDecision
    track_id: int
    face_id: Optional[str]
    user_id: Optional[str]
    name: Optional[str]
    confidence: float
    reason: str
    value: str = field(init=False)
    
    def __post_init__(self):
        self.value = self.decision.value


class GateController:
    """
    Controls the physical gate relay via GPIO.
    Handles timing for gate open duration.
    """
    
    def __init__(
        self,
        gpio_enabled: bool = False,
        relay_pin: int = 17,
        active_low: bool = True,
        open_duration: float = 5.0,
        cooldown: float = 2.0
    ):
        self.gpio_enabled = gpio_enabled
        self.relay_pin = relay_pin
        self.active_low = active_low
        self.open_duration = open_duration
        self.cooldown = cooldown
        
        self._gpio = None
        self._lock = threading.Lock()
        self._is_open = False
        self._last_open_time = 0.0
        self._close_timer: Optional[threading.Timer] = None
        self._initialized = False
        
        # Stats
        self._stats = {
            "total_opens": 0,
            "authorized_opens": 0,
            "wanted_opens": 0,
            "rejected_unknown": 0,
        }
        
        # Callback for state changes
        self.on_state_change: Optional[Callable[[bool], None]] = None
    
    @property
    def state(self) -> GateState:
        """Get current gate state."""
        return GateState.OPEN if self._is_open else GateState.CLOSED
    
    def initialize(self) -> bool:
        """Initialize GPIO for relay control."""
        if self._initialized:
            return True
        
        if not self.gpio_enabled:
            logger.info("GPIO disabled - gate control in simulation mode")
            self._initialized = True
            return True
        
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.relay_pin, GPIO.OUT)
            
            # Set initial state (closed)
            initial = GPIO.HIGH if self.active_low else GPIO.LOW
            GPIO.output(self.relay_pin, initial)
            
            logger.info(f"GPIO initialized on pin {self.relay_pin}")
            self._initialized = True
            return True
            
        except ImportError:
            logger.warning("RPi.GPIO not available - running in simulation mode")
            self.gpio_enabled = False
            self._initialized = True
            return True
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            return False
    
    def open_gate(
        self,
        decision: GateAction = None,
        person_id: str = None,
        track_id: int = None,
        confidence: float = 0.0
    ) -> bool:
        """
        Open the gate for configured duration.
        If gate is already open, EXTENDS the open duration (resets timer).
        Returns True if gate was opened/extended.
        """
        with self._lock:
            now = time.time()
            
            # Track stats based on decision
            if decision:
                self._stats["total_opens"] += 1
                if decision == GateAction.OPEN or (hasattr(decision, 'value') and decision.value == "AUTHORIZED"):
                    self._stats["authorized_opens"] += 1
                elif hasattr(decision, 'value') and decision.value == "WANTED":
                    self._stats["wanted_opens"] += 1
            
            # Cancel any pending close timer (ALWAYS - to extend duration)
            if self._close_timer:
                self._close_timer.cancel()
                self._close_timer = None
            
            # If gate is already open, just extend the timer
            if self._is_open:
                # Reset the close timer (extend open duration)
                self._close_timer = threading.Timer(self.open_duration, self._auto_close)
                self._close_timer.start()
                self._last_open_time = now
                
                logger.info(f"Gate EXTENDED (will close in {self.open_duration}s)")
                return True
            
            # Open gate (was closed)
            self._set_relay(True)
            self._is_open = True
            self._last_open_time = now
            
            # Schedule close
            self._close_timer = threading.Timer(self.open_duration, self._auto_close)
            self._close_timer.start()
            
            logger.info(f"Gate OPENED (will close in {self.open_duration}s)")
            
            if self.on_state_change:
                self.on_state_change(True)
            
            return True
    
    def reject(self, track_id: int = None):
        """Record a rejection (unknown person)."""
        with self._lock:
            self._stats["rejected_unknown"] += 1
    
    def close_gate(self):
        """Immediately close the gate."""
        with self._lock:
            if self._close_timer:
                self._close_timer.cancel()
                self._close_timer = None
            
            self._set_relay(False)
            self._is_open = False
            
            logger.info("Gate CLOSED")
            
            if self.on_state_change:
                self.on_state_change(False)
    
    def _auto_close(self):
        """Auto-close callback from timer."""
        with self._lock:
            self._set_relay(False)
            self._is_open = False
            self._close_timer = None
            
            logger.info("Gate auto-closed")
            
            if self.on_state_change:
                self.on_state_change(False)
    
    def _set_relay(self, open_state: bool):
        """Set relay GPIO state."""
        if self._gpio and self.gpio_enabled:
            if self.active_low:
                # Active low: LOW = relay on = gate open
                value = self._gpio.LOW if open_state else self._gpio.HIGH
            else:
                # Active high: HIGH = relay on = gate open
                value = self._gpio.HIGH if open_state else self._gpio.LOW
            
            self._gpio.output(self.relay_pin, value)
    
    def is_open(self) -> bool:
        """Check if gate is currently open."""
        with self._lock:
            return self._is_open
    
    def get_stats(self) -> dict:
        """Get gate operation statistics."""
        with self._lock:
            return self._stats.copy()
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self._close_timer:
            self._close_timer.cancel()
        
        if self._gpio and self.gpio_enabled:
            try:
                self._gpio.cleanup(self.relay_pin)
            except Exception as e:
                logger.error(f"GPIO cleanup error: {e}")
            self._set_relay(False)
            self._is_open = False
            self._close_timer = None
            
            logger.info("Gate auto-closed")
            
            if self.on_state_change:
                self.on_state_change(False)
    
    def _set_relay(self, open_state: bool):
        """Set relay GPIO state."""
        if self._gpio and self.gpio_enabled:
            if self.active_low:
                # Active low: LOW = relay on = gate open
                value = self._gpio.LOW if open_state else self._gpio.HIGH
            else:
                # Active high: HIGH = relay on = gate open
                value = self._gpio.HIGH if open_state else self._gpio.LOW
            
            self._gpio.output(self.relay_pin, value)
    
    def is_open(self) -> bool:
        """Check if gate is currently open."""
        with self._lock:
            return self._is_open
    
    def cleanup(self):
        """Cleanup GPIO resources."""
        if self._close_timer:
            self._close_timer.cancel()
        
        if self._gpio and self.gpio_enabled:
            try:
                self._gpio.cleanup(self.relay_pin)
            except Exception as e:
                logger.error(f"GPIO cleanup error: {e}")


class DecisionEngine:
    """
    Makes access decisions based on recognition results.
    
    Decision logic:
    - AUTHORIZED: Open gate (if confidence above threshold)
    - UNKNOWN: Keep closed (deny access)
    - WANTED: Open gate (to capture/detain), trigger alert
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        wanted_confidence_threshold: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.wanted_confidence_threshold = wanted_confidence_threshold
    
    def make_decision(
        self,
        match_found: bool,
        person_id: Optional[str] = None,
        confidence: float = 0.0,
        status: str = "AUTHORIZED",
    ) -> GateDecision:
        """
        Make access decision based on recognition result.
        
        Args:
            match_found: Whether a match was found in the database
            person_id: ID of the matched person (if found)
            confidence: Recognition confidence score
            status: Status from database (AUTHORIZED, WANTED)
        
        Returns:
            GateDecision enum value
        """
        if not match_found:
            logger.info(f"ACCESS DENIED: No match found")
            return GateDecision.UNKNOWN
        
        # Check if wanted
        if status.upper() == "WANTED":
            if confidence >= self.wanted_confidence_threshold:
                logger.warning(f"⚠️ WANTED DETECTED: {person_id} (confidence: {confidence:.2f})")
                return GateDecision.WANTED
            else:
                logger.info(f"Low confidence wanted match: {person_id} ({confidence:.2f})")
                return GateDecision.UNKNOWN
        
        # Check if authorized
        if status.upper() == "AUTHORIZED":
            if confidence >= self.confidence_threshold:
                logger.info(f"ACCESS GRANTED: {person_id} (confidence: {confidence:.2f})")
                return GateDecision.AUTHORIZED
            else:
                logger.info(f"Low confidence match: {person_id} ({confidence:.2f})")
                return GateDecision.UNKNOWN
        
        # Default to unknown
        logger.info(f"Unknown status '{status}' for {person_id}")
        return GateDecision.UNKNOWN
