"""
Alarm System for Gate Access Control.

Provides audio alerts for different security events:
- WANTED: Hard alarm (continuous beep)
- UNKNOWN: Soft alarm (single beep)
- AUTHORIZED: Optional soft notification

Platform Support:
- Windows: winsound.Beep
- Linux/Mac: System beep or audio file playback
- Raspberry Pi: GPIO buzzer or audio

Thread-safe and non-blocking.
"""

import threading
import time
import logging
import os
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class AlarmType(Enum):
    """Types of alarms."""
    WANTED = "wanted"       # Hard alarm - continuous/loud
    UNKNOWN = "unknown"     # Soft alarm - single beep
    AUTHORIZED = "authorized"  # Optional notification
    SILENT = "silent"       # No sound


@dataclass
class AlarmConfig:
    """Configuration for alarm sounds."""
    # Wanted alarm (hard)
    wanted_frequency: int = 2500      # Hz
    wanted_duration: int = 500        # ms per beep
    wanted_beeps: int = 5             # Number of beeps
    wanted_gap: int = 100             # ms between beeps
    
    # Unknown alarm (soft)
    unknown_frequency: int = 1500     # Hz
    unknown_duration: int = 300       # ms
    unknown_beeps: int = 2
    unknown_gap: int = 150
    
    # Authorized notification (optional)
    authorized_frequency: int = 800   # Hz
    authorized_duration: int = 100    # ms
    authorized_enabled: bool = False
    
    # Global settings
    enabled: bool = True
    volume: float = 1.0               # 0.0 to 1.0 (for audio file playback)
    cooldown_seconds: float = 5.0     # Min time between same alarm type


class AlarmSystem:
    """
    Cross-platform alarm system.
    
    Usage:
        alarm = AlarmSystem()
        alarm.trigger(AlarmType.WANTED, person_name="John Doe")
        alarm.trigger(AlarmType.UNKNOWN)
    """
    
    def __init__(self, config: Optional[AlarmConfig] = None):
        self.config = config or AlarmConfig()
        self._lock = threading.Lock()
        self._last_alarm: dict = {}  # alarm_type -> timestamp
        self._alarm_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Platform detection
        self._platform = self._detect_platform()
        self._beep_func = self._get_beep_function()
        
        logger.info(f"AlarmSystem initialized for platform: {self._platform}")
    
    def _detect_platform(self) -> str:
        """Detect the current platform."""
        if os.name == 'nt':
            return 'windows'
        elif os.path.exists('/proc/cpuinfo'):
            # Check if running on Raspberry Pi
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    if 'Raspberry Pi' in f.read():
                        return 'raspberry_pi'
            except:
                pass
            return 'linux'
        else:
            return 'other'
    
    def _get_beep_function(self) -> Optional[Callable]:
        """Get platform-specific beep function."""
        if self._platform == 'windows':
            try:
                import winsound
                return lambda freq, dur: winsound.Beep(freq, dur)
            except ImportError:
                logger.warning("winsound not available on Windows")
                return None
        
        elif self._platform == 'raspberry_pi':
            # GPIO buzzer support (if available)
            try:
                import RPi.GPIO as GPIO
                # We'll use PWM for buzzer
                return self._raspberry_pi_beep
            except ImportError:
                logger.info("RPi.GPIO not available, trying system beep")
                return self._linux_beep
        
        elif self._platform == 'linux':
            return self._linux_beep
        
        return None
    
    def _linux_beep(self, frequency: int, duration: int):
        """Linux system beep fallback."""
        try:
            # Try using 'beep' command
            os.system(f'beep -f {frequency} -l {duration} 2>/dev/null || '
                     f'echo -e "\\a" 2>/dev/null')
        except:
            pass
    
    def _raspberry_pi_beep(self, frequency: int, duration: int):
        """Raspberry Pi GPIO buzzer beep."""
        # This would need GPIO setup - simplified version
        try:
            import RPi.GPIO as GPIO
            # Assuming buzzer on GPIO 18 (PWM0)
            BUZZER_PIN = 18
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUZZER_PIN, GPIO.OUT)
            
            pwm = GPIO.PWM(BUZZER_PIN, frequency)
            pwm.start(50)  # 50% duty cycle
            time.sleep(duration / 1000.0)
            pwm.stop()
        except Exception as e:
            logger.debug(f"GPIO beep failed: {e}")
            self._linux_beep(frequency, duration)
    
    def trigger(
        self,
        alarm_type: AlarmType,
        person_name: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """
        Trigger an alarm.
        
        Args:
            alarm_type: Type of alarm to trigger
            person_name: Optional name for logging
            force: Bypass cooldown check
        
        Returns:
            True if alarm was triggered, False if skipped (cooldown/disabled)
        """
        if not self.config.enabled:
            return False
        
        if alarm_type == AlarmType.SILENT:
            return False
        
        # Check cooldown
        now = time.time()
        with self._lock:
            last_time = self._last_alarm.get(alarm_type, 0)
            if not force and (now - last_time) < self.config.cooldown_seconds:
                logger.debug(f"Alarm {alarm_type.value} skipped (cooldown)")
                return False
            self._last_alarm[alarm_type] = now
        
        # Log the alarm
        name_str = f" ({person_name})" if person_name else ""
        logger.warning(f"ALARM TRIGGERED: {alarm_type.value.upper()}{name_str}")
        
        # Play alarm in background thread (non-blocking)
        self._stop_event.clear()
        self._alarm_thread = threading.Thread(
            target=self._play_alarm,
            args=(alarm_type,),
            daemon=True,
            name=f"Alarm-{alarm_type.value}"
        )
        self._alarm_thread.start()
        
        return True
    
    def _play_alarm(self, alarm_type: AlarmType):
        """Play alarm sound (runs in background thread)."""
        if self._beep_func is None:
            logger.warning("No beep function available")
            return
        
        try:
            if alarm_type == AlarmType.WANTED:
                # Hard alarm - multiple loud beeps
                for _ in range(self.config.wanted_beeps):
                    if self._stop_event.is_set():
                        break
                    self._beep_func(
                        self.config.wanted_frequency,
                        self.config.wanted_duration
                    )
                    time.sleep(self.config.wanted_gap / 1000.0)
            
            elif alarm_type == AlarmType.UNKNOWN:
                # Soft alarm - single/double beep
                for _ in range(self.config.unknown_beeps):
                    if self._stop_event.is_set():
                        break
                    self._beep_func(
                        self.config.unknown_frequency,
                        self.config.unknown_duration
                    )
                    time.sleep(self.config.unknown_gap / 1000.0)
            
            elif alarm_type == AlarmType.AUTHORIZED and self.config.authorized_enabled:
                # Optional soft notification
                self._beep_func(
                    self.config.authorized_frequency,
                    self.config.authorized_duration
                )
        
        except Exception as e:
            logger.error(f"Alarm playback error: {e}")
    
    def stop(self):
        """Stop any currently playing alarm."""
        self._stop_event.set()
        if self._alarm_thread and self._alarm_thread.is_alive():
            self._alarm_thread.join(timeout=1.0)
    
    def set_enabled(self, enabled: bool):
        """Enable or disable alarm system."""
        self.config.enabled = enabled
        logger.info(f"Alarm system {'enabled' if enabled else 'disabled'}")
    
    def test(self):
        """Test all alarm types."""
        logger.info("Testing alarm system...")
        
        # Test wanted
        self.trigger(AlarmType.WANTED, "Test", force=True)
        time.sleep(3)
        
        # Test unknown
        self.trigger(AlarmType.UNKNOWN, force=True)
        time.sleep(1)
        
        # Test authorized (if enabled)
        if self.config.authorized_enabled:
            self.trigger(AlarmType.AUTHORIZED, force=True)
        
        logger.info("Alarm test complete")


# Global instance
_alarm_system: Optional[AlarmSystem] = None
_alarm_lock = threading.Lock()


def get_alarm_system() -> AlarmSystem:
    """Get global alarm system instance."""
    global _alarm_system
    with _alarm_lock:
        if _alarm_system is None:
            _alarm_system = AlarmSystem()
        return _alarm_system


def trigger_alarm(alarm_type: AlarmType, person_name: Optional[str] = None) -> bool:
    """Convenience function to trigger alarm."""
    return get_alarm_system().trigger(alarm_type, person_name)
