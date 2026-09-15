"""Engine-agnostic, UI-agnostic runtime.

Nothing in this package may import ``textual`` or a vendor SDK. That rule is
what keeps a headless CLI, a TUI, and the test suite all driving one core.
"""

from agentdeck.core.agent import Agent, AgentSpec
from agentdeck.core.events import (
    Event,
    ReasoningDelta,
    RunFailed,
    RunFinished,
    RunStarted,
    TextDelta,
    Usage,
)
from agentdeck.core.messages import Message, Role, assistant, system, user
from agentdeck.core.orchestrator import Orchestrator, Pattern, merge
from agentdeck.core.session import Session

__all__ = [
    "Agent",
    "AgentSpec",
    "Event",
    "Message",
    "Orchestrator",
    "Pattern",
    "ReasoningDelta",
    "Role",
    "RunFailed",
    "RunFinished",
    "RunStarted",
    "Session",
    "TextDelta",
    "Usage",
    "assistant",
    "merge",
    "system",
    "user",
]
