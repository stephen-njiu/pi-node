"""
Gate Controller and Decision Engine.
Handles gate hardware control and access decisions.
"""

import threading
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import logging


logger = logging.getLogger(__name__)


class AccessStatus(Enum):
    """Face recognition status."""
    AUTHORIZED = "AUTHORIZED"
    UNKNOWN = "UNKNOWN"
    WANTED = "WANTED"


class GateAction(Enum):
    """Gate action to take."""
    OPEN = "OPEN"
    CLOSE = "CLOSE"  # Keep closed / deny


@dataclass
class AccessDecision:
    """Result of access decision."""
    action: GateAction
    status: AccessStatus
    track_id: int
    face_id: Optional[str]
    user_id: Optional[str]
    name: Optional[str]
    confidence: float
    reason: str


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
        
        # Callback for state changes
        self.on_state_change: Optional[Callable[[bool], None]] = None
        
        self._init_gpio()
    
    def _init_gpio(self):
        """Initialize GPIO for relay control."""
        if not self.gpio_enabled:
            logger.info("GPIO disabled - gate control in simulation mode")
            return
        
        try:
            import RPi.GPIO as GPIO
            self._gpio = GPIO
            
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.relay_pin, GPIO.OUT)
            
            # Set initial state (closed)
            initial = GPIO.HIGH if self.active_low else GPIO.LOW
            GPIO.output(self.relay_pin, initial)
            
            logger.info(f"GPIO initialized on pin {self.relay_pin}")
            
        except ImportError:
            logger.warning("RPi.GPIO not available - running in simulation mode")
            self.gpio_enabled = False
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")
            self.gpio_enabled = False
    
    def open_gate(self) -> bool:
        """
        Open the gate for configured duration.
        Returns True if gate was opened, False if in cooldown.
        """
        with self._lock:
            now = time.time()
            
            # Check cooldown
            if now - self._last_open_time < self.cooldown and self._is_open:
                logger.debug("Gate already open, ignoring")
                return False
            
            # Cancel any pending close timer
            if self._close_timer:
                self._close_timer.cancel()
            
            # Open gate
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
    - AUTHORIZED: Open gate
    - UNKNOWN: Keep closed (deny access)
    - WANTED: Open gate (to capture/detain), trigger alert
    """
    
    def __init__(
        self,
        gate_controller: GateController,
        on_alert: Optional[Callable[[AccessDecision], None]] = None
    ):
        self.gate_controller = gate_controller
        self.on_alert = on_alert  # Callback for UNKNOWN/WANTED alerts
    
    def decide(
        self,
        track_id: int,
        status: str,
        face_id: Optional[str] = None,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        confidence: float = 0.0
    ) -> AccessDecision:
        """
        Make access decision based on recognition result.
        
        Args:
            track_id: Tracker ID
            status: Recognition status (AUTHORIZED, UNKNOWN, WANTED)
            face_id: Matched face ID
            user_id: Matched user ID
            name: Matched user name
            confidence: Recognition confidence
        
        Returns:
            AccessDecision with action and details
        """
        try:
            access_status = AccessStatus(status)
        except ValueError:
            access_status = AccessStatus.UNKNOWN
        
        decision = None
        
        if access_status == AccessStatus.AUTHORIZED:
            # Grant access
            decision = AccessDecision(
                action=GateAction.OPEN,
                status=access_status,
                track_id=track_id,
                face_id=face_id,
                user_id=user_id,
                name=name,
                confidence=confidence,
                reason=f"Authorized user: {name or 'Unknown'}"
            )
            self.gate_controller.open_gate()
            logger.info(f"ACCESS GRANTED: {name} (confidence: {confidence:.2f})")
            
        elif access_status == AccessStatus.WANTED:
            # Open gate to capture (security protocol)
            # But also trigger alert
            decision = AccessDecision(
                action=GateAction.OPEN,
                status=access_status,
                track_id=track_id,
                face_id=face_id,
                user_id=user_id,
                name=name,
                confidence=confidence,
                reason=f"WANTED individual detected: {name or 'Unknown'}"
            )
            self.gate_controller.open_gate()
            logger.warning(f"⚠️ WANTED DETECTED: {name} - Gate opened for capture")
            
            # Trigger alert
            if self.on_alert:
                self.on_alert(decision)
            
        else:  # UNKNOWN
            # Deny access
            decision = AccessDecision(
                action=GateAction.CLOSE,
                status=access_status,
                track_id=track_id,
                face_id=None,
                user_id=None,
                name=None,
                confidence=confidence,
                reason="Unknown individual - access denied"
            )
            logger.info(f"ACCESS DENIED: Unknown face (confidence: {confidence:.2f})")
            
            # Trigger alert for unknown
            if self.on_alert:
                self.on_alert(decision)
        
        return decision
    
    def get_gate_status(self) -> str:
        """Get current gate status."""
        return "OPEN" if self.gate_controller.is_open() else "CLOSED"
