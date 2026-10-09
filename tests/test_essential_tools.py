from datetime import UTC, datetime, timedelta, timezone

import pytest

from zordon.tools.essentials import calculate, get_current_datetime


def test_current_datetime_reports_injected_local_and_utc_values() -> None:
    local = datetime(
        2026,
        8,
        11,
        8,
        15,
        30,
        tzinfo=timezone(timedelta(hours=-7), "PDT"),
    )

    result = get_current_datetime(lambda: local)

    assert result.ok is True
    assert result.data == {
        "local_iso": "2026-08-11T08:15:30-07:00",
        "utc_iso": "2026-08-11T15:15:30+00:00",
        "utc_offset": "-07:00",
        "timezone_name": "PDT",
    }


def test_current_datetime_requires_an_aware_clock_value() -> None:
    naive = datetime(2026, 8, 11, 8, 15, 30, tzinfo=UTC).replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        get_current_datetime(lambda: naive)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", "3"),
        ("7 - 10", "-3"),
        ("(48 * 17) / 3", "272"),
        ("7 // 2", "3"),
        ("7 % 4", "3"),
        ("2 ** -3", "0.125"),
        ("-(2 + 3)", "-5"),
        ("+(2)", "2"),
        ("0.1 + 0.2", "0.3"),
        ("1.2300 + 0", "1.23"),
        ("-0", "0"),
        ("1e2 + 0.5", "100.5"),
    ],
)
def test_calculator_supported_operations(expression: str, expected: str) -> None:
    result = calculate(expression)

    assert result.ok is True
    assert result.data["result"] == expected


@pytest.mark.parametrize(
    "expression",
    [
        "abs(1)",
        "number + 1",
        "(1).real",
        "[1][0]",
        "True",
        "1 < 2",
        "[1, 2]",
        "(1, 2)",
        "{1, 2}",
        "{'one': 1}",
        "1 & 1",
        "1 | 1",
        "1 ^ 1",
        "1 << 1",
        "1 >> 1",
        "~1",
        "'1'",
        "1j",
    ],
)
def test_calculator_rejects_unsupported_syntax(expression: str) -> None:
    assert_invalid_calculation(expression)


@pytest.mark.parametrize("expression", ["2 ** 101", "2 ** -101", "2 ** 0.5"])
def test_calculator_rejects_invalid_exponents(expression: str) -> None:
    assert_invalid_calculation(expression)


@pytest.mark.parametrize("operator", ["/", "//", "%"])
def test_calculator_rejects_zero_division(operator: str) -> None:
    assert_invalid_calculation(f"1 {operator} 0")


def test_calculator_rejects_expression_over_256_characters() -> None:
    assert_invalid_calculation("1" + " " * 256)


def test_calculator_rejects_expression_over_64_ast_nodes() -> None:
    assert_invalid_calculation(" + ".join(["1"] * 33))


@pytest.mark.parametrize(
    "expression",
    [
        "(10 ** 100) ** 100",
        "1e999999 * 10",
    ],
)
def test_calculator_rejects_nonfinite_or_oversized_output(expression: str) -> None:
    assert_invalid_calculation(expression)


@pytest.mark.parametrize("expression", ["", "1 +", "("])
def test_calculator_rejects_malformed_syntax(expression: str) -> None:
    assert_invalid_calculation(expression)


def assert_invalid_calculation(expression: str) -> None:
    result = calculate(expression)
    assert result.ok is False
    assert result.code == "calculation_invalid"
    assert result.data == {}
