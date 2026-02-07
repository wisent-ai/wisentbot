"""Tests for SkillProfileManager."""
import pytest
from singularity.skill_profiles import SkillProfileManager, PROFILES, ProfileDefinition


class TestSkillProfiles:
    def setup_method(self):
        SkillProfileManager.clear_cache()
        SkillProfileManager.clear_custom_profiles()

    def test_builtin_profiles_exist(self):
        assert "minimal" in PROFILES
        assert "developer" in PROFILES
        assert "social" in PROFILES
        assert "full" in PROFILES
        assert "autonomous" in PROFILES

    def test_list_profiles(self):
        profiles = SkillProfileManager.list_profiles()
        assert len(profiles) >= 9
        names = [p["name"] for p in profiles]
        assert "developer" in names
        assert "minimal" in names

    def test_get_profile(self):
        p = SkillProfileManager.get_profile("minimal")
        assert p is not None
        assert p.name == "minimal"
        assert "filesystem" in p.skills
        assert "shell" in p.skills

    def test_resolve_skill_ids_single(self):
        ids = SkillProfileManager.resolve_skill_ids("developer")
        assert "filesystem" in ids
        assert "shell" in ids
        assert "github" in ids

    def test_resolve_skill_ids_combined(self):
        ids = SkillProfileManager.resolve_skill_ids(["minimal", "social"])
        assert "filesystem" in ids
        assert "shell" in ids
        assert "twitter" in ids
        # No duplicates
        assert len(ids) == len(set(ids))

    def test_resolve_skill_ids_unknown(self):
        ids = SkillProfileManager.resolve_skill_ids("nonexistent")
        assert ids == []

    def test_resolve_raw_skill_id(self):
        ids = SkillProfileManager.resolve_skill_ids("filesystem")
        assert ids == ["filesystem"]

    def test_register_custom_profile(self):
        SkillProfileManager.register_profile("test_profile", {
            "description": "Test",
            "skills": ["filesystem", "shell"],
            "tags": ["test"],
        })
        p = SkillProfileManager.get_profile("test_profile")
        assert p is not None
        assert p.skills == ["filesystem", "shell"]

    def test_profile_inheritance(self):
        SkillProfileManager.register_profile("extended", {
            "description": "Extended minimal",
            "skills": ["github"],
            "extends": "minimal",
        })
        p = SkillProfileManager.get_profile("extended")
        assert "github" in p.skills
        assert "filesystem" in p.skills
        assert "shell" in p.skills

    def test_get_skill_classes(self):
        classes = SkillProfileManager.get_skill_classes("minimal")
        assert len(classes) == 2
        class_names = [c.__name__ for c in classes]
        assert "FilesystemSkill" in class_names
        assert "ShellSkill" in class_names

    def test_find_profiles_with_skill(self):
        profiles = SkillProfileManager.find_profiles_with_skill("filesystem")
        assert "minimal" in profiles
        assert "developer" in profiles
        assert "full" in profiles

    def test_find_profiles_by_tag(self):
        profiles = SkillProfileManager.find_profiles_by_tag("lightweight")
        assert "minimal" in profiles

    def test_suggest_profile(self):
        assert SkillProfileManager.suggest_profile("write code and fix bugs") == "developer"
        assert SkillProfileManager.suggest_profile("post a tweet") == "social"
        assert SkillProfileManager.suggest_profile("deploy to vercel") == "web"
        assert SkillProfileManager.suggest_profile("self improve and learn") == "autonomous"
        # Unknown defaults to developer
        assert SkillProfileManager.suggest_profile("xyz random") == "developer"

    def test_full_profile_has_all_skills(self):
        available = SkillProfileManager.get_available_skill_ids()
        full_profile = SkillProfileManager.get_profile("full")
        assert set(full_profile.skills) == set(available)
