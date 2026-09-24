"""Model-facing exclusive range choices; source runtime retains its existing contract."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

_RANGE_FIELDS = ("range_mode", "start_at", "end_at", "history_days", "history_months")

class _ClosedRange(BaseModel):
    model_config = ConfigDict(extra="forbid")

class AutomaticRange(_ClosedRange):
    kind: Literal["auto"]

class AllHistoryRange(_ClosedRange):
    kind: Literal["all_history"]

class CalendarRange(_ClosedRange):
    kind: Literal["calendar"]
    period: Literal["day", "week", "month"]

class RelativeRange(_ClosedRange):
    kind: Literal["relative"]
    amount: int = Field(ge=1, strict=True)
    unit: Literal["days", "months"]

    @field_validator("amount", mode="before")
    @classmethod
    def whole_number_amount(cls, value):
        # A whole amount may arrive as 30.0 or "30"; bool, fractions and null still fail the strict int.
        if (isinstance(value, float) and value.is_integer()) or (isinstance(value, str) and value.isdecimal()):
            return int(value)
        return value

class BetweenRange(_ClosedRange):
    kind: Literal["between"]
    start_at: str = Field(min_length=1)
    end_at: str = Field(min_length=1)

_RANGE = TypeAdapter(Annotated[AutomaticRange | AllHistoryRange | CalendarRange | RelativeRange | BetweenRange, Field(discriminator="kind")])

def health_range_schema():
    schema = _RANGE.json_schema()
    definitions = schema["$defs"]
    # Local variants keep unrelated operation fields out of this choice.
    variants = [definitions[item["$ref"].rsplit("/", 1)[-1]] for item in schema["oneOf"]]
    for variant in variants:
        kind = variant["properties"]["kind"]
        kind["enum"] = [kind.pop("const")]
    return {"oneOf": variants, "description": 'Choose one complete date selector. Examples: {"kind":"calendar","period":"day"}, {"kind":"relative","amount":7,"unit":"days"}, {"kind":"between","start_at":"2026-05-17","end_at":"2026-08-17"}, {"kind":"all_history"}, {"kind":"auto"}. Calendar requires period (day/week/month), relative requires amount and unit (days/months), between requires start_at and end_at. Calendar uses the trusted request clock. Paging never changes this scope; purpose selects which observations to return.'}

def normalize_health_ranges(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("health_reads"), list):
        return payload
    converted = dict(payload)
    converted["health_reads"] = []
    for index, original in enumerate(payload["health_reads"]):
        if not isinstance(original, dict) or "range" not in original:
            converted["health_reads"].append(original)
            continue
        operation = dict(original)
        try:
            selection = _RANGE.validate_python(operation.pop("range"))
        except ValidationError as error:
            # Locate the failure in the payload, as the batch contract locates its own errors.
            raise ValidationError.from_exception_data(error.title, [
                {"type": item["type"], "loc": ("health_reads", index, "range", *item["loc"]), "input": item["input"],
                 **({"ctx": item["ctx"]} if "ctx" in item else {})}
                for item in error.errors()]) from error
        import logging
        logging.getLogger(__name__).info(
            "manager_health_range kind=%s period=%s", selection.kind,
            selection.period if selection.kind == "calendar" else "none")
        # Do not accept two selectors or silently discard a conflicting field.
        if any(field in operation for field in _RANGE_FIELDS):
            operation["range"] = original["range"]
        elif selection.kind == "all_history":
            operation["range_mode"] = "all"
        elif selection.kind == "calendar":
            operation["range_mode"] = "current_" + selection.period
        elif selection.kind == "relative":
            operation["range_mode"] = "auto"
            operation["history_" + selection.unit] = selection.amount
        elif selection.kind == "between":
            operation.update(range_mode="between", start_at=selection.start_at, end_at=selection.end_at)
        else:
            operation["range_mode"] = "auto"
        converted["health_reads"].append(operation)
    return converted
