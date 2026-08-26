"""Trusted service-to-service context for a D10 Agent turn."""

from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator


class D10RequestContext(BaseModel):
    """Identity and correlation data asserted by D10, never by the model."""

    model_config = ConfigDict(extra="forbid")

    clinic_id: str = Field(..., min_length=1, max_length=256)
    user_id: str = Field(..., min_length=1, max_length=256)
    conversation_id: str = Field(..., min_length=1, max_length=256)
    actor_id: str = Field(..., min_length=1, max_length=256)
    actor_role: str = Field(..., min_length=1, max_length=64)
    timezone: str = Field(..., min_length=1, max_length=128)
    source_message_id: str = Field(..., min_length=1, max_length=512)
    correlation_id: str = Field(..., min_length=1, max_length=256)
    causation_id: Optional[str] = Field(default=None, max_length=256)
    reservation_id: Optional[str] = Field(default=None, max_length=256)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be a valid IANA timezone") from exc
        return value

    def tool_envelope(self) -> Dict[str, Any]:
        """Return the exact immutable context sent to every D10 tool call."""
        return self.model_dump(exclude_none=True)

    def model_context(self) -> Dict[str, str]:
        """Expose only context useful for reasoning, not internal identifiers."""
        return {
            "actor_role": self.actor_role,
            "user_timezone": self.timezone,
        }
