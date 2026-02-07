#!/usr/bin/env python3
"""
Skill Profiles - Predefined skill sets for purpose-specific agents.

Instead of loading ALL skills at startup (heavy, many credentials needed),
agents can load a specific profile of skills matching their purpose.

Profiles serve multiple pillars:
- Replication: spawn lightweight agents with only needed capabilities
- Revenue: deploy targeted service agents (e.g., code-only, social-only)
- Self-Improvement: agent can switch profiles based on current task
- Goal Setting: profiles constrain agent behavior to relevant actions

Usage:
    from singularity.skill_profiles import SkillProfileManager, PROFILES

    # List available profiles
    profiles = SkillProfileManager.list_profiles()

    # Get skill classes for a profile
    classes = SkillProfileManager.get_skill_classes("developer")

    # Combine profiles
    classes = SkillProfileManager.get_skill_classes(["developer", "social"])

    # Custom profile
    SkillProfileManager.register_profile("my_profile", {
        "description": "My custom agent",
        "skills": ["filesystem", "shell", "github"],
    })
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type, Union

# Lazy imports - we only import skill classes when actually needed
_SKILL_CLASS_REGISTRY: Dict[str, str] = {
    # skill_id -> (module_path, class_name)
    "content": ("singularity.skills.content", "ContentCreationSkill"),
    "twitter": ("singularity.skills.twitter", "TwitterSkill"),
    "github": ("singularity.skills.github", "GitHubSkill"),
    "namecheap": ("singularity.skills.namecheap", "NamecheapSkill"),
    "email": ("singularity.skills.email", "EmailSkill"),
    "browser": ("singularity.skills.browser", "BrowserSkill"),
    "vercel": ("singularity.skills.vercel", "VercelSkill"),
    "filesystem": ("singularity.skills.filesystem", "FilesystemSkill"),
    "shell": ("singularity.skills.shell", "ShellSkill"),
    "mcp_client": ("singularity.skills.mcp_client", "MCPClientSkill"),
    "request": ("singularity.skills.request", "RequestSkill"),
    "self_modify": ("singularity.skills.self_modify", "SelfModifySkill"),
    "steering": ("singularity.skills.steering", "SteeringSkill"),
    "memory": ("singularity.skills.memory", "MemorySkill"),
    "orchestrator": ("singularity.skills.orchestrator", "OrchestratorSkill"),
    "crypto": ("singularity.skills.crypto", "CryptoSkill"),
}


@dataclass
class ProfileDefinition:
    """Definition of a skill profile."""
    name: str
    description: str
    skills: List[str]  # List of skill IDs
    tags: List[str] = field(default_factory=list)
    extends: Optional[str] = None  # Profile to inherit from


# Built-in profiles
PROFILES: Dict[str, ProfileDefinition] = {
    "minimal": ProfileDefinition(
        name="minimal",
        description="Bare minimum for basic file and shell operations",
        skills=["filesystem", "shell"],
        tags=["lightweight", "local"],
    ),
    "developer": ProfileDefinition(
        name="developer",
        description="Software development agent: code, git, shell, web requests",
        skills=["filesystem", "shell", "github", "request", "content"],
        tags=["code", "git", "development"],
    ),
    "social": ProfileDefinition(
        name="social",
        description="Social media and content agent: Twitter, email, content creation",
        skills=["twitter", "email", "content", "browser"],
        tags=["social", "marketing", "outreach"],
    ),
    "web": ProfileDefinition(
        name="web",
        description="Web operations: browser, HTTP requests, Vercel deployment",
        skills=["browser", "request", "vercel", "filesystem"],
        tags=["web", "deployment", "http"],
    ),
    "infrastructure": ProfileDefinition(
        name="infrastructure",
        description="Infrastructure management: domains, deployment, crypto",
        skills=["namecheap", "vercel", "shell", "filesystem", "crypto"],
        tags=["infra", "devops", "deployment"],
    ),
    "autonomous": ProfileDefinition(
        name="autonomous",
        description="Self-improving agent: self-modification, memory, orchestration",
        skills=["self_modify", "memory", "orchestrator", "steering", "filesystem", "shell"],
        tags=["self-improvement", "autonomous", "meta"],
    ),
    "full": ProfileDefinition(
        name="full",
        description="All available skills loaded (original behavior)",
        skills=list(_SKILL_CLASS_REGISTRY.keys()),
        tags=["full", "all"],
    ),
    "service": ProfileDefinition(
        name="service",
        description="Service-oriented agent: content creation, requests, email for client work",
        skills=["content", "filesystem", "shell", "request", "email", "github"],
        tags=["revenue", "service", "client"],
    ),
    "research": ProfileDefinition(
        name="research",
        description="Research agent: browser, requests, filesystem for data gathering",
        skills=["browser", "request", "filesystem", "shell", "content", "memory"],
        tags=["research", "data", "analysis"],
    ),
}


class SkillProfileManager:
    """Manages skill profiles and resolves skill classes."""

    _custom_profiles: Dict[str, ProfileDefinition] = {}
    _import_cache: Dict[str, type] = {}

    @classmethod
    def list_profiles(cls) -> List[Dict]:
        """List all available profiles with descriptions."""
        all_profiles = {**PROFILES, **cls._custom_profiles}
        return [
            {
                "name": p.name,
                "description": p.description,
                "skills": p.skills,
                "skill_count": len(p.skills),
                "tags": p.tags,
            }
            for p in all_profiles.values()
        ]

    @classmethod
    def get_profile(cls, name: str) -> Optional[ProfileDefinition]:
        """Get a profile by name."""
        if name in cls._custom_profiles:
            return cls._custom_profiles[name]
        return PROFILES.get(name)

    @classmethod
    def register_profile(cls, name: str, definition: dict) -> ProfileDefinition:
        """
        Register a custom profile.

        Args:
            name: Profile name
            definition: Dict with 'description', 'skills', optional 'tags' and 'extends'

        Returns:
            The created ProfileDefinition
        """
        skills = list(definition.get("skills", []))

        # Handle profile inheritance
        extends = definition.get("extends")
        if extends:
            parent = cls.get_profile(extends)
            if parent:
                # Merge parent skills with child skills (child wins on duplicates)
                parent_skills = set(parent.skills)
                child_skills = set(skills)
                skills = list(parent_skills | child_skills)

        profile = ProfileDefinition(
            name=name,
            description=definition.get("description", f"Custom profile: {name}"),
            skills=skills,
            tags=definition.get("tags", ["custom"]),
            extends=extends,
        )
        cls._custom_profiles[name] = profile
        return profile

    @classmethod
    def resolve_skill_ids(cls, profile: Union[str, List[str]]) -> List[str]:
        """
        Resolve a profile name (or list of names) to a list of skill IDs.

        Args:
            profile: Profile name, list of profile names, or list of skill IDs

        Returns:
            Deduplicated list of skill IDs
        """
        if isinstance(profile, str):
            prof = cls.get_profile(profile)
            if prof:
                return list(prof.skills)
            # If not a profile name, treat as a single skill ID
            if profile in _SKILL_CLASS_REGISTRY:
                return [profile]
            return []

        # List: combine multiple profiles/skill IDs
        seen: Set[str] = set()
        result: List[str] = []
        for item in profile:
            ids = cls.resolve_skill_ids(item)
            for sid in ids:
                if sid not in seen:
                    seen.add(sid)
                    result.append(sid)
        return result

    @classmethod
    def _import_skill_class(cls, skill_id: str) -> Optional[type]:
        """Import a skill class by its ID. Caches imports."""
        if skill_id in cls._import_cache:
            return cls._import_cache[skill_id]

        entry = _SKILL_CLASS_REGISTRY.get(skill_id)
        if not entry:
            return None

        module_path, class_name = entry
        try:
            import importlib
            module = importlib.import_module(module_path)
            klass = getattr(module, class_name)
            cls._import_cache[skill_id] = klass
            return klass
        except (ImportError, AttributeError):
            return None

    @classmethod
    def get_skill_classes(cls, profile: Union[str, List[str]]) -> List[type]:
        """
        Get skill classes for a profile (or combined profiles).

        Args:
            profile: Profile name, list of profile names, or mixed list

        Returns:
            List of skill classes ready for instantiation
        """
        skill_ids = cls.resolve_skill_ids(profile)
        classes = []
        for sid in skill_ids:
            klass = cls._import_skill_class(sid)
            if klass is not None:
                classes.append(klass)
        return classes

    @classmethod
    def get_available_skill_ids(cls) -> List[str]:
        """List all known skill IDs."""
        return list(_SKILL_CLASS_REGISTRY.keys())

    @classmethod
    def find_profiles_with_skill(cls, skill_id: str) -> List[str]:
        """Find all profiles that include a given skill."""
        all_profiles = {**PROFILES, **cls._custom_profiles}
        return [
            name for name, prof in all_profiles.items()
            if skill_id in prof.skills
        ]

    @classmethod
    def find_profiles_by_tag(cls, tag: str) -> List[str]:
        """Find profiles that have a given tag."""
        all_profiles = {**PROFILES, **cls._custom_profiles}
        return [
            name for name, prof in all_profiles.items()
            if tag in prof.tags
        ]

    @classmethod
    def suggest_profile(cls, task_description: str) -> str:
        """
        Suggest the best profile for a task based on keyword matching.

        Args:
            task_description: Description of what the agent needs to do

        Returns:
            Profile name that best matches the task
        """
        task_lower = task_description.lower()

        # Keyword -> profile mapping
        keyword_scores: Dict[str, int] = {}
        keywords = {
            "minimal": ["simple", "basic", "file", "minimal", "lightweight"],
            "developer": ["code", "program", "develop", "git", "commit", "repo",
                         "python", "javascript", "debug", "test", "build", "pr",
                         "pull request", "branch", "merge"],
            "social": ["tweet", "post", "social", "twitter", "email", "market",
                      "promote", "outreach", "announce", "share"],
            "web": ["web", "browser", "scrape", "deploy", "website", "http",
                   "url", "page", "vercel", "html"],
            "infrastructure": ["domain", "dns", "deploy", "server", "crypto",
                              "blockchain", "infra", "devops"],
            "autonomous": ["self", "improve", "learn", "modify", "evolve",
                          "memory", "orchestrate", "spawn", "replicate"],
            "service": ["client", "service", "revenue", "work", "deliver",
                       "project", "invoice"],
            "research": ["research", "analyze", "data", "gather", "investigate",
                        "study", "report", "find"],
        }

        for profile, words in keywords.items():
            score = sum(1 for w in words if w in task_lower)
            keyword_scores[profile] = score

        # Return the highest scoring profile, default to "developer"
        best = max(keyword_scores, key=keyword_scores.get)
        if keyword_scores[best] == 0:
            return "developer"
        return best

    @classmethod
    def clear_cache(cls):
        """Clear the import cache (useful for testing)."""
        cls._import_cache.clear()

    @classmethod
    def clear_custom_profiles(cls):
        """Clear all custom profiles (useful for testing)."""
        cls._custom_profiles.clear()
