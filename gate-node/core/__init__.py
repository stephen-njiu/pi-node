"""Core module for gate control and state management."""

from .track_state import TrackStateManager
from .gate_control import GateController, DecisionEngine, AccessStatus, GateAction, AccessDecision

# Aliases for compatibility
TrackStatus = AccessStatus
GateDecision = AccessDecision


def create_gate_controller_from_config(config) -> GateController:
    """Factory function to create GateController from config object."""
    return GateController(
        gpio_enabled=config.GPIO_ENABLED,
        relay_pin=config.GPIO_RELAY_PIN,
        active_low=config.GPIO_ACTIVE_LOW,
        open_duration=config.GATE_OPEN_DURATION,
        cooldown=config.GATE_COOLDOWN
    )


__all__ = [
    "TrackStateManager",
    "TrackStatus",
    "GateController",
    "GateDecision",
    "DecisionEngine",
    "AccessStatus",
    "GateAction",
    "AccessDecision",
    "create_gate_controller_from_config",
]
