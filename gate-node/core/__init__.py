"""Core module for gate control and state management."""

from .track_state import TrackStateManager, TrackStatus, TrackState
from .gate_control import (
    GateController,
    DecisionEngine,
    GateDecision,
    GateState,
    GateAction,
    AccessDecision,
)


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
    "TrackState",
    "GateController",
    "GateDecision",
    "GateState",
    "DecisionEngine",
    "GateAction",
    "AccessDecision",
    "create_gate_controller_from_config",
]
