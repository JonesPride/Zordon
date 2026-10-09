from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Context, Decimal, DecimalException, localcontext
from typing import Final

from zordon.tools.base import ToolResult

_MAX_EXPRESSION_LENGTH: Final = 256
_MAX_AST_NODES: Final = 64
_MAX_EXPONENT: Final = 100
_MAX_RESULT_LENGTH: Final = 1_000
_DECIMAL_CONTEXT: Final = Context(prec=50)

_BINARY_OPERATORS: Final = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS: Final = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


class _InvalidCalculation(ValueError):
    pass


def get_current_datetime(
    clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
) -> ToolResult:
    """Return local and UTC timestamps from an injectable system clock."""
    local = clock()
    if not isinstance(local, datetime) or local.tzinfo is None:
        raise ValueError("The clock must return a timezone-aware datetime.")
    offset = local.utcoffset()
    if offset is None:
        raise ValueError("The clock must return a timezone-aware datetime.")

    return ToolResult.success(
        "Current date and time retrieved.",
        {
            "local_iso": local.isoformat(),
            "utc_iso": local.astimezone(UTC).isoformat(),
            "utc_offset": _format_utc_offset(offset.total_seconds()),
            "timezone_name": local.tzname() or "",
        },
    )


def calculate(expression: str) -> ToolResult:
    """Evaluate a bounded arithmetic expression using decimal arithmetic."""
    try:
        result = _calculate(expression)
    except (DecimalException, _InvalidCalculation, SyntaxError, ValueError):
        return ToolResult.failure(
            "calculation_invalid",
            "The calculation is invalid or exceeds the allowed limits.",
        )
    return ToolResult.success("Calculation completed.", {"result": result})


def _calculate(expression: str) -> str:
    if not isinstance(expression, str):
        raise _InvalidCalculation
    if not expression.strip() or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise _InvalidCalculation

    tree = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > _MAX_AST_NODES:
        raise _InvalidCalculation

    with localcontext(_DECIMAL_CONTEXT):
        result = _evaluate(tree, expression)
    if not result.is_finite():
        raise _InvalidCalculation

    formatted = _format_decimal(result)
    if len(formatted) > _MAX_RESULT_LENGTH:
        raise _InvalidCalculation
    return formatted


def _evaluate(node: ast.AST, source: str) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, source)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise _InvalidCalculation
        segment = ast.get_source_segment(source, node)
        if segment is None:
            raise _InvalidCalculation
        try:
            return Decimal(segment)
        except DecimalException as exc:
            raise _InvalidCalculation from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
        operand = _evaluate(node.operand, source)
        return _UNARY_OPERATORS[type(node.op)](operand)
    if isinstance(node, ast.BinOp):
        left = _evaluate(node.left, source)
        right = _evaluate(node.right, source)
        if isinstance(node.op, ast.Pow):
            if right != right.to_integral_value() or not (-_MAX_EXPONENT <= right <= _MAX_EXPONENT):
                raise _InvalidCalculation
            return left ** int(right)
        operation = _BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise _InvalidCalculation
        return operation(left, right)
    raise _InvalidCalculation


def _format_decimal(value: Decimal) -> str:
    if value.is_zero():
        return "0"
    digits = len(value.as_tuple().digits)
    exponent = value.as_tuple().exponent
    if not isinstance(exponent, int):
        raise _InvalidCalculation
    if exponent >= 0:
        fixed_length = digits + exponent
    elif digits + exponent > 0:
        fixed_length = digits + 1
    else:
        fixed_length = 2 - exponent
    if value.is_signed():
        fixed_length += 1
    if fixed_length > _MAX_RESULT_LENGTH:
        raise _InvalidCalculation

    formatted = format(value, "f")
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _format_utc_offset(total_seconds: float) -> str:
    sign = "+" if total_seconds >= 0 else "-"
    total_minutes = abs(int(total_seconds)) // 60
    hours, minutes = divmod(total_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"
