"""Tests for the FilesystemSkill - file operations that run locally."""

from pathlib import Path

import pytest

from singularity.skills.filesystem import FilesystemSkill


@pytest.fixture
def fs_skill(tmp_dir):
    """Create a FilesystemSkill rooted at a temporary directory."""
    return FilesystemSkill(base_path=str(tmp_dir))


@pytest.fixture
def populated_dir(tmp_dir):
    """Create a temporary directory with some test files."""
    # Create directory structure
    (tmp_dir / "src").mkdir()
    (tmp_dir / "src" / "main.py").write_text("print('hello world')\n")
    (tmp_dir / "src" / "utils.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef multiply(a, b):\n    return a * b\n"
    )
    (tmp_dir / "README.md").write_text("# Test Project\n\nThis is a test.\n")
    (tmp_dir / "data").mkdir()
    (tmp_dir / "data" / "config.json").write_text('{"key": "value"}\n')
    return tmp_dir


# ── Write ───────────────────────────────────────────────────────────────


class TestWrite:
    @pytest.mark.asyncio
    async def test_write_new_file(self, fs_skill, tmp_dir):
        result = await fs_skill.execute("write", {"path": "test.txt", "content": "Hello!"})
        assert result.success is True
        assert (tmp_dir / "test.txt").read_text() == "Hello!"

    @pytest.mark.asyncio
    async def test_write_creates_parent_dirs(self, fs_skill, tmp_dir):
        result = await fs_skill.execute(
            "write", {"path": "deep/nested/file.txt", "content": "Nested content"}
        )
        assert result.success is True
        assert (tmp_dir / "deep" / "nested" / "file.txt").read_text() == "Nested content"

    @pytest.mark.asyncio
    async def test_write_overwrite_existing(self, fs_skill, tmp_dir):
        (tmp_dir / "existing.txt").write_text("old content")
        result = await fs_skill.execute(
            "write", {"path": "existing.txt", "content": "new content"}
        )
        assert result.success is True
        assert (tmp_dir / "existing.txt").read_text() == "new content"

    @pytest.mark.asyncio
    async def test_write_reports_bytes(self, fs_skill):
        content = "Hello, World!"
        result = await fs_skill.execute("write", {"path": "size.txt", "content": content})
        assert result.success is True
        assert result.data["bytes"] == len(content)


# ── View ────────────────────────────────────────────────────────────────


class TestView:
    @pytest.mark.asyncio
    async def test_view_file(self, fs_skill, populated_dir):
        result = await fs_skill.execute("view", {"path": "README.md"})
        assert result.success is True
        assert "# Test Project" in result.data["content"]
        assert result.data["total_lines"] == 3

    @pytest.mark.asyncio
    async def test_view_with_offset(self, fs_skill, populated_dir):
        result = await fs_skill.execute("view", {"path": "src/utils.py", "offset": 1})
        assert result.success is True
        # Should skip the first line "def add(a, b):"
        assert "return a + b" in result.data["content"]
        assert result.data["offset"] == 1

    @pytest.mark.asyncio
    async def test_view_with_limit(self, fs_skill, populated_dir):
        result = await fs_skill.execute("view", {"path": "src/utils.py", "limit": 2})
        assert result.success is True
        assert result.data["lines_returned"] == 2

    @pytest.mark.asyncio
    async def test_view_nonexistent_file(self, fs_skill):
        result = await fs_skill.execute("view", {"path": "nonexistent.txt"})
        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_view_directory_fails(self, fs_skill, populated_dir):
        result = await fs_skill.execute("view", {"path": "src"})
        assert result.success is False
        assert "not a file" in result.message.lower()


# ── Ls ──────────────────────────────────────────────────────────────────


class TestLs:
    @pytest.mark.asyncio
    async def test_ls_root(self, fs_skill, populated_dir):
        result = await fs_skill.execute("ls", {"path": "."})
        assert result.success is True
        names = [e["name"] for e in result.data["entries"]]
        assert "README.md" in names
        assert "src" in names
        assert "data" in names

    @pytest.mark.asyncio
    async def test_ls_subdirectory(self, fs_skill, populated_dir):
        result = await fs_skill.execute("ls", {"path": "src"})
        assert result.success is True
        names = [e["name"] for e in result.data["entries"]]
        assert "main.py" in names
        assert "utils.py" in names

    @pytest.mark.asyncio
    async def test_ls_entries_have_type(self, fs_skill, populated_dir):
        result = await fs_skill.execute("ls", {"path": "."})
        entries = {e["name"]: e for e in result.data["entries"]}
        assert entries["src"]["type"] == "dir"
        assert entries["README.md"]["type"] == "file"

    @pytest.mark.asyncio
    async def test_ls_nonexistent_path(self, fs_skill):
        result = await fs_skill.execute("ls", {"path": "nonexistent"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_ls_file_path_fails(self, fs_skill, populated_dir):
        result = await fs_skill.execute("ls", {"path": "README.md"})
        assert result.success is False


# ── Glob ────────────────────────────────────────────────────────────────


class TestGlob:
    @pytest.mark.asyncio
    async def test_glob_all_python_files(self, fs_skill, populated_dir):
        result = await fs_skill.execute("glob", {"pattern": "**/*.py"})
        assert result.success is True
        files = result.data["files"]
        # Should find main.py and utils.py
        py_files = [f for f in files if f.endswith(".py")]
        assert len(py_files) >= 2

    @pytest.mark.asyncio
    async def test_glob_specific_pattern(self, fs_skill, populated_dir):
        result = await fs_skill.execute("glob", {"pattern": "*.md"})
        assert result.success is True
        files = result.data["files"]
        assert any("README.md" in f for f in files)

    @pytest.mark.asyncio
    async def test_glob_no_matches(self, fs_skill, populated_dir):
        result = await fs_skill.execute("glob", {"pattern": "*.xyz"})
        assert result.success is True
        assert len(result.data["files"]) == 0


# ── Grep ────────────────────────────────────────────────────────────────


class TestGrep:
    @pytest.mark.asyncio
    async def test_grep_in_file(self, fs_skill, populated_dir):
        result = await fs_skill.execute(
            "grep", {"pattern": "hello", "path": "src/main.py"}
        )
        assert result.success is True
        assert len(result.data["matches"]) >= 1

    @pytest.mark.asyncio
    async def test_grep_in_directory(self, fs_skill, populated_dir):
        result = await fs_skill.execute(
            "grep", {"pattern": "def ", "path": "src", "include": "*.py"}
        )
        assert result.success is True
        # Should find "def add" and "def multiply" in utils.py
        assert len(result.data["matches"]) >= 2

    @pytest.mark.asyncio
    async def test_grep_case_insensitive(self, fs_skill, populated_dir):
        result = await fs_skill.execute(
            "grep", {"pattern": "HELLO", "path": "src/main.py"}
        )
        assert result.success is True
        assert len(result.data["matches"]) >= 1

    @pytest.mark.asyncio
    async def test_grep_no_matches(self, fs_skill, populated_dir):
        result = await fs_skill.execute(
            "grep", {"pattern": "nonexistent_string_xyz", "path": "src"}
        )
        assert result.success is True
        assert len(result.data["matches"]) == 0

    @pytest.mark.asyncio
    async def test_grep_invalid_regex(self, fs_skill, populated_dir):
        result = await fs_skill.execute(
            "grep", {"pattern": "[invalid", "path": "src"}
        )
        assert result.success is False
        assert "regex" in result.message.lower()


# ── Mkdir ───────────────────────────────────────────────────────────────


class TestMkdir:
    @pytest.mark.asyncio
    async def test_mkdir_simple(self, fs_skill, tmp_dir):
        result = await fs_skill.execute("mkdir", {"path": "new_dir"})
        assert result.success is True
        assert (tmp_dir / "new_dir").is_dir()

    @pytest.mark.asyncio
    async def test_mkdir_nested(self, fs_skill, tmp_dir):
        result = await fs_skill.execute("mkdir", {"path": "a/b/c"})
        assert result.success is True
        assert (tmp_dir / "a" / "b" / "c").is_dir()

    @pytest.mark.asyncio
    async def test_mkdir_existing_dir(self, fs_skill, populated_dir):
        result = await fs_skill.execute("mkdir", {"path": "src"})
        assert result.success is True  # exist_ok=True


# ── Rm ──────────────────────────────────────────────────────────────────


class TestRm:
    @pytest.mark.asyncio
    async def test_rm_file(self, fs_skill, populated_dir, tmp_dir):
        assert (tmp_dir / "README.md").exists()
        result = await fs_skill.execute("rm", {"path": "README.md"})
        assert result.success is True
        assert not (tmp_dir / "README.md").exists()

    @pytest.mark.asyncio
    async def test_rm_empty_directory(self, fs_skill, tmp_dir):
        (tmp_dir / "empty_dir").mkdir()
        result = await fs_skill.execute("rm", {"path": "empty_dir"})
        assert result.success is True
        assert not (tmp_dir / "empty_dir").exists()

    @pytest.mark.asyncio
    async def test_rm_nonempty_dir_without_recursive_fails(self, fs_skill, populated_dir):
        result = await fs_skill.execute("rm", {"path": "src"})
        assert result.success is False

    @pytest.mark.asyncio
    async def test_rm_nonempty_dir_recursive(self, fs_skill, populated_dir, tmp_dir):
        result = await fs_skill.execute("rm", {"path": "src", "recursive": True})
        assert result.success is True
        assert not (tmp_dir / "src").exists()

    @pytest.mark.asyncio
    async def test_rm_nonexistent(self, fs_skill):
        result = await fs_skill.execute("rm", {"path": "nonexistent"})
        assert result.success is False


# ── Unknown action ──────────────────────────────────────────────────────


class TestUnknownAction:
    @pytest.mark.asyncio
    async def test_unknown_action(self, fs_skill):
        result = await fs_skill.execute("nonexistent_action", {})
        assert result.success is False
        assert "Unknown action" in result.message


# ── Path resolution ─────────────────────────────────────────────────────


class TestPathResolution:
    def test_relative_path(self, fs_skill, tmp_dir):
        resolved = fs_skill._resolve_path("test.txt")
        assert resolved == tmp_dir / "test.txt"

    def test_absolute_path(self, fs_skill):
        resolved = fs_skill._resolve_path("/absolute/path")
        assert resolved == Path("/absolute/path")

    def test_check_credentials(self, fs_skill):
        assert fs_skill.check_credentials() is True

    def test_manifest(self, fs_skill):
        manifest = fs_skill.manifest
        assert manifest.skill_id == "filesystem"
        assert len(manifest.actions) > 0
        assert manifest.required_credentials == []
