from typing import Any, Union


class NodeReference:
    """Node parameter type"""

    ref: str

    def __init__(self, ref: str) -> None:
        self.ref = ref

    def unwrap(self):
        return self.ref


class Array:
    """Node parameter type of `list[str | float | int]`"""

    ref: list[str | float | int]

    def __init__(self, ref: list[str | float | int]) -> None:
        self.ref = ref

    def unwrap(self):
        return self.ref


def format_param_value(value: Any, value_type: str):
    match value_type:
        case "Array":
            s = str(value)
            parsed_value = parse_array(s, [str, float, int], "list[int | float | str]")
            return Array(parsed_value)
        case "float":
            return float(value)
        case "int":
            return int(value)
        case "bool":
            return bool(value)
        case "NodeReference":
            return NodeReference(str(value))
        case "list[str]":
            return parse_array(str(value), [str], "list[str]")
        case "list[float]":
            return parse_array(str(value), [float], "list[float]")
        case "list[int]":
            return parse_array(str(value), [int], "list[int]")
        case "select" | "str":
            return str(value)
        case _:
            return value


def parse_array(
    str_value: str, type_list: list[Any], param_type: str
) -> list[Union[int, float, str]]:
    if not str_value:
        return []

    val_list = [val.strip() for val in str_value.split(",")]
    # First try to cast into int, then float, then keep as string if all else fails
    for t in type_list:
        try:
            return list(map(t, val_list))
        except ValueError:
            continue

    raise ValueError(
        f"Couldn't parse list items with type {','.join([str(t) for t in type_list])}."
        + f"Value should be comma (',') separated {' | '.join([t.__name__ for t in type_list])} for parameter type {param_type}."
    )
