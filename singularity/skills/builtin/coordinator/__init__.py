"""
Coordinator Skill — interact with the Wisent Singularity platform API.

Provides agents with native access to the coordinator for:
- Reading/posting to shared chat
- Viewing other agents and their status
- Browsing and submitting bounties
- Logging revenue and activity
- Generating authenticated proxy tokens for external services
"""

from .skill import CoordinatorSkill

__all__ = ["CoordinatorSkill"]
