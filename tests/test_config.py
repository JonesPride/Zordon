import pytest

from zordon.config import ConfigurationError, load_settings


def test_load_settings_uses_tier_one_defaults() -> None:
    settings = load_settings({"OPENAI_API_KEY": "test-key"})

    assert settings.api_key == "test-key"
    assert settings.model == "gpt-5.6-terra"
    assert settings.timeout_seconds == 60.0


def test_load_settings_accepts_supported_overrides() -> None:
    settings = load_settings(
        {
            "OPENAI_API_KEY": "test-key",
            "ZORDON_MODEL": "gpt-5.6-sol",
            "ZORDON_REQUEST_TIMEOUT_SECONDS": "15.5",
        }
    )

    assert settings.model == "gpt-5.6-sol"
    assert settings.timeout_seconds == 15.5


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
        (
            {
                "OPENAI_API_KEY": "test-key",
                "ZORDON_REQUEST_TIMEOUT_SECONDS": "0",
            },
            "positive number",
        ),
    ],
)
def test_load_settings_rejects_invalid_configuration(
    environ: dict[str, str], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        load_settings(environ)


def test_missing_key_error_explains_secure_powershell_setup() -> None:
    with pytest.raises(ConfigurationError) as error:
        load_settings({})

    message = str(error.value)
    assert "https://platform.openai.com/api-keys" in message
    assert (
        "if (-not (Test-Path .env)) { Copy-Item .env.example .env }"
        in message
    )
    assert "text editor" in message
    assert "command history" in message


def test_settings_repr_does_not_reveal_api_key() -> None:
    settings = load_settings({"OPENAI_API_KEY": "super-secret-test-key"})

    assert "super-secret-test-key" not in repr(settings)
