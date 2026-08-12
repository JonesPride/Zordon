from pathlib import Path

import pytest

from zordon.config import ApprovedRoot, ConfigurationError, Tier2Limits, load_settings


def test_load_settings_uses_tier_one_defaults() -> None:
    settings = load_settings({"OPENAI_API_KEY": "test-key"})

    assert settings.api_key == "test-key"
    assert settings.model == "gpt-5.6-terra"
    assert settings.timeout_seconds == 60.0
    assert settings.history_message_limit == 40
    assert settings.output_token_limit == 2048
    assert settings.reasoning_effort == "medium"
    assert settings.debug_log_path is None


def test_load_settings_accepts_supported_overrides() -> None:
    settings = load_settings(
        {
            "OPENAI_API_KEY": "test-key",
            "ZORDON_MODEL": "gpt-5.6-sol",
            "ZORDON_REQUEST_TIMEOUT_SECONDS": "15.5",
            "ZORDON_HISTORY_MESSAGE_LIMIT": "12",
            "ZORDON_OUTPUT_TOKEN_LIMIT": "900",
            "ZORDON_REASONING_EFFORT": "high",
            "ZORDON_DEBUG_LOG": "logs/zordon-debug.jsonl",
        }
    )

    assert settings.model == "gpt-5.6-sol"
    assert settings.timeout_seconds == 15.5
    assert settings.history_message_limit == 12
    assert settings.output_token_limit == 900
    assert settings.reasoning_effort == "high"
    assert settings.debug_log_path is not None
    assert settings.debug_log_path.parts[-2:] == ("logs", "zordon-debug.jsonl")


@pytest.mark.parametrize(
    ("environ", "message"),
    [
        ({}, "OPENAI_API_KEY"),
        ({"OPENAI_API_KEY": "   "}, "OPENAI_API_KEY"),
        (
            {
                "OPENAI_API_KEY": "test-key",
                "ZORDON_REQUEST_TIMEOUT_SECONDS": "zero",
            },
            "positive number",
        ),
        ({"OPENAI_API_KEY": "test-key", "ZORDON_HISTORY_MESSAGE_LIMIT": "3"}, "even integer"),
        ({"OPENAI_API_KEY": "test-key", "ZORDON_OUTPUT_TOKEN_LIMIT": "0"}, "between 1 and"),
        ({"OPENAI_API_KEY": "test-key", "ZORDON_REASONING_EFFORT": "extreme"}, "one of"),
        (
            {
                "OPENAI_API_KEY": "test-key",
                "ZORDON_REQUEST_TIMEOUT_SECONDS": "0",
            },
            "positive number",
        ),
    ],
)
def test_load_settings_rejects_invalid_configuration(environ: dict[str, str], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(environ)


def test_missing_key_error_explains_secure_powershell_setup() -> None:
    with pytest.raises(ConfigurationError) as error:
        load_settings({})

    message = str(error.value)
    assert "https://platform.openai.com/api-keys" in message
    assert "if (-not (Test-Path .env)) { Copy-Item .env.example .env }" in message
    assert "text editor" in message
    assert "command history" in message


def test_settings_repr_does_not_reveal_api_key() -> None:
    settings = load_settings({"OPENAI_API_KEY": "super-secret-test-key"})

    assert "super-secret-test-key" not in repr(settings)


def test_load_settings_has_no_approved_roots_by_default() -> None:
    settings = load_settings({"OPENAI_API_KEY": "test-key"})

    assert settings.approved_roots == ()
    assert settings.tier2_limits == Tier2Limits()


def test_load_settings_parses_approved_roots_case_insensitively(tmp_path: Path) -> None:
    notes = tmp_path / "Notes"
    projects = tmp_path / "Projects"
    notes.mkdir()
    projects.mkdir()

    settings = load_settings(
        {
            "OPENAI_API_KEY": "test-key",
            "ZORDON_FOLDER_PROJECTS": str(projects),
            "ZORDON_FOLDER_NOTES": str(notes),
        }
    )

    assert settings.approved_roots == (
        ApprovedRoot("notes", notes.resolve()),
        ApprovedRoot("projects", projects.resolve()),
    )
    assert "test-key" not in repr(settings)


@pytest.mark.parametrize("suffix", ["", "1NOTES", "notes", "BAD-NAME", "A" * 33])
def test_load_settings_rejects_invalid_folder_suffixes(tmp_path: Path, suffix: str) -> None:
    with pytest.raises(ConfigurationError, match="approved folder"):
        load_settings({"OPENAI_API_KEY": "test-key", f"ZORDON_FOLDER_{suffix}": str(tmp_path)})


def test_load_settings_rejects_missing_and_file_roots(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    file_root = tmp_path / "file.txt"
    file_root.write_text("not a directory")

    for path in (missing, file_root):
        with pytest.raises(ConfigurationError, match="approved folder"):
            load_settings({"OPENAI_API_KEY": "test-key", "ZORDON_FOLDER_X": str(path)})


def test_load_settings_rejects_duplicate_ids_and_canonical_roots(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    class DuplicateItems(dict[str, str]):
        def items(self):  # type: ignore[override]
            return [
                ("OPENAI_API_KEY", "test-key"),
                ("ZORDON_FOLDER_NOTES", str(root)),
                ("ZORDON_FOLDER_NOTES", str(root)),
            ]

    with pytest.raises(ConfigurationError, match="Duplicate approved folder ID"):
        load_settings(DuplicateItems(OPENAI_API_KEY="test-key"))

    with pytest.raises(ConfigurationError, match="same approved folder"):
        load_settings(
            {
                "OPENAI_API_KEY": "test-key",
                "ZORDON_FOLDER_NOTES": str(root),
                "ZORDON_FOLDER_OTHER": str(root / "."),
            }
        )


@pytest.mark.parametrize(
    ("key", "minimum", "maximum", "attribute"),
    [
        ("ZORDON_DOCUMENT_SCAN_LIMIT", 100, 50_000, "document_scan"),
        ("ZORDON_AUDIO_SCAN_LIMIT", 100, 100_000, "audio_scan"),
        ("ZORDON_SEARCH_RESULT_LIMIT", 1, 25, "search_results"),
        ("ZORDON_DOCUMENT_READ_CHARS", 1_000, 20_000, "document_read_chars"),
    ],
)
def test_load_settings_accepts_limit_boundaries(
    key: str, minimum: int, maximum: int, attribute: str
) -> None:
    for value in (minimum, maximum):
        settings = load_settings({"OPENAI_API_KEY": "test-key", key: str(value)})
        assert getattr(settings.tier2_limits, attribute) == value


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ZORDON_DOCUMENT_SCAN_LIMIT", "99"),
        ("ZORDON_DOCUMENT_SCAN_LIMIT", "50001"),
        ("ZORDON_AUDIO_SCAN_LIMIT", "99"),
        ("ZORDON_AUDIO_SCAN_LIMIT", "100001"),
        ("ZORDON_SEARCH_RESULT_LIMIT", "0"),
        ("ZORDON_SEARCH_RESULT_LIMIT", "26"),
        ("ZORDON_DOCUMENT_READ_CHARS", "999"),
        ("ZORDON_DOCUMENT_READ_CHARS", "20001"),
        ("ZORDON_DOCUMENT_SCAN_LIMIT", "many"),
    ],
)
def test_load_settings_rejects_out_of_range_limits(key: str, value: str) -> None:
    with pytest.raises(ConfigurationError, match=key):
        load_settings({"OPENAI_API_KEY": "test-key", key: value})
