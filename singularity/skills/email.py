#!/usr/bin/env python3
"""Email skill for sending emails via multiple providers."""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from .base import Skill, SkillManifest

logger = logging.getLogger(__name__)


@dataclass
class EmailSkill(Skill):
    """Send emails via Resend, SendGrid, or SMTP."""

    manifest: SkillManifest = field(default_factory=lambda: SkillManifest(
        name="email",
        description="Send emails via Resend, SendGrid, or SMTP",
        version="0.1.0",
        required_credentials=[],
        capabilities=["send_email"],
    ))

    def check_credentials(self) -> dict:
        """Check which email provider credentials are available."""
        providers = {}

        # Resend
        if os.getenv("RESEND_API_KEY"):
            providers["resend"] = True

        # SendGrid
        if os.getenv("SENDGRID_API_KEY"):
            providers["sendgrid"] = True

        # SMTP
        if os.getenv("SMTP_HOST") and os.getenv("SMTP_USER"):
            providers["smtp"] = True

        return providers

    async def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        from_email: Optional[str] = None,
        provider: