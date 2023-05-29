import typing
import numpy as np
import pandas as pd
from box import Box, box_list
import plotly.graph_objects as go  # type:ignore
from typing import Union, Literal, get_args, Any, cast


__all__ = ["DataContainer"]

DCType = Literal[
    "grayscale",
    "matrix",
    "dataframe",
    "image",
    "ordered_pair",
    "ordered_triple",
    "scalar",
    "plotly",
    "parametric_grayscale",
    "parametric_matrix",
    "parametric_dataframe",
    "parametric_image",
    "parametric_ordered_pair",
    "parametric_ordered_triple",
    "parametric_scalar",
    "parametric_plotly",
]

DCNpArrayType = np.ndarray[Union[int, float], np.dtype[Any]]
DCKwargsValue = Union[
    list[Union[int, float]],
    int,
    float,
    dict[str, Union[int, float, DCNpArrayType]],
    DCNpArrayType,
    pd.DataFrame,
    go.Figure,
    None,
]


class DataContainer(Box):
    """
    A class that processes various types of data and supports dot assignment

    Learn more: https://github.com/flojoy-io/flojoy-python/issues/4

    Usage
    -----
    import numpy as np

    v = DataContainer()

    v.x = np.linspace(1,20,0.1)

    v.y = np.sin(v.x)

    v.type = 'ordered_pair'

    """

    allowed_types = typing.get_args(DCType)
    allowed_keys = ["x", "y", "z", "t", "m", "c", "r", "g", "b", "a", "fig"]
    combinations = {
        "x": ["y", "t", "z", "fig"],
        "y": ["x", "t", "z", "fig"],
        "z": ["x", "y", "t", "fig"],
        "c": ["t", "fig"],
        "m": ["t", "fig"],
        "t": [*(value for value in allowed_keys if value not in ["t"])],
        "r": ["g", "b", "t", "a", "fig"],
        "g": ["r", "b", "t", "a", "fig"],
        "b": ["r", "g", "t", "a", "fig"],
        "a": ["r", "g", "b", "t", "fig"],
        "fig": [*(k for k in allowed_keys if k not in ["fig"])],
    }

    type: DCType

    def copy(self):
        # Create an instance of DataContainer class
        copied_instance = DataContainer(**self)
        return copied_instance

    def _ndarrayify(
        self, value: DCKwargsValue
    ) -> Union[DCNpArrayType, pd.DataFrame, dict[str, DCNpArrayType], go.Figure, None]:
        if isinstance(value, int) or isinstance(value, float):
            return np.array([value])
        elif isinstance(value, dict):
            arrayified_value: dict[str, DCNpArrayType] = {}
            for k, v in value.items():
                arrayified_value[k] = cast(DCNpArrayType, self._ndarrayify(v))
            return arrayified_value
        elif isinstance(value, box_list.BoxList):
            arrayified_value: dict[str, DCNpArrayType] = {}
            for k, v in value.__dict__.items():
                arrayified_value[k] = cast(DCNpArrayType, self._ndarrayify(v))
            return arrayified_value
        elif isinstance(value, pd.DataFrame):
            return value
        elif isinstance(value, np.ndarray):
            return value
        elif isinstance(value, list):
            return np.array(value)
        elif isinstance(value, go.Figure):
            return value
        elif value is None:
            return value
        else:
            raise ValueError(
                f"DataContainer keys must be any of following types: {get_args(DCKwargsValue)}"
            )

    def __validate_key_for_type(self, data_type: DCType, key: str):
        match data_type:
            case "ordered_pair":
                if key not in ["x", "y"]:
                    raise KeyError(self.__build_error_text(key, data_type))
            case "grayscale" | "matrix" | "dataframe":
                if key not in ["m"]:
                    raise KeyError(self.__build_error_text(key, data_type))
            case "image":
                if key not in ["r", "g", "b", "a"]:
                    raise KeyError(self.__build_error_text(key, data_type))
            case "ordered_triple":
                if key not in ["x", "y", "z"]:
                    raise KeyError(self.__build_error_text(key, data_type))
            case "scalar":
                if key not in ["c"]:
                    raise KeyError(self.__build_error_text(key, data_type))
            case "plotly":
                if key not in ["fig", *(k for k in self.combinations["fig"])]:
                    raise KeyError(self.__build_error_text(key, data_type))
            case _:
                if data_type.startswith("parametric_") and key != "t":
                    splitted_type = cast(DCType, data_type.split("parametric_")[1])
                    self.__validate_key_for_type(splitted_type, key)

    def __init__(  # type:ignore
        self, type: DCType = "ordered_pair", **kwargs: DCKwargsValue
    ):
        self.type = type
        for k, v in kwargs.items():
            self[k] = v

    def __getitem__(self, key: str, _ignore_default: bool = False) -> Any:
        return super().__getitem__(key, _ignore_default)  # type: ignore

    def __setitem__(self, key: str, value: DCKwargsValue) -> None:
        if key != "type":
            formatted_value = self._ndarrayify(value)
            super().__setitem__(key, formatted_value)  # type:ignore
        else:
            super().__setitem__(key, value)  # type: ignore

    def __check_combination(self, key: str, keys: list[str], allowed_keys: list[str]):
        for i in keys:
            if i not in allowed_keys:
                raise ValueError(f"You cant have {key} with {i}")

    def __check_for_missing_keys(self, dc_type: DCType, keys: list[str]):
        match dc_type:
            case "grayscale" | "matrix" | "dataframe":
                if "m" not in keys:
                    raise KeyError(f'm key must be provided for type "{dc_type}"')
            case "image":
                if "r" and "g" and "b" and "a" not in keys:
                    raise KeyError(
                        f'r g b a keys must be provided for type "{dc_type}"'
                    )
            case "ordered_pair":
                if "x" and "y" not in keys:
                    raise KeyError(f'x and y keys must be provided for "{dc_type}"')

            case "ordered_triple":
                if "x" and "y" and "z" not in keys:
                    raise KeyError(f'x, y and z keys must be provided for "{dc_type}"')

            case "scalar":
                if "c" not in keys:
                    raise KeyError(f'c key must be provided for type "{dc_type}"')

            case "plotly":
                if "fig" not in keys:
                    raise KeyError(f'f key must be provided for type "{dc_type}"')

            case _:
                if dc_type.startswith("parametric_"):
                    if "t" not in keys:
                        raise KeyError(f't key must be provided for "{dc_type}"')
                    t = self["t"]
                    is_ascending_order = all(
                        t[i] <= t[i + 1] for i in range(len(t) - 1)
                    )
                    if is_ascending_order is not True:
                        raise ValueError("t key must be in ascending order")
                else:
                    raise ValueError(f'Invalid data type "{dc_type}"')

    def __build_error_text(self, key: str, data_type: str):
        return f'Invalid key "{key}" provided for data type "{data_type}"'

    def validate(self):
        dc_type = self.type
        dc_keys = list(cast(list[str], self.keys()))
        for k in dc_keys:
            if k != "type":
                self.__check_combination(
                    k,
                    list(key for key in dc_keys if key not in ["type", k]),
                    self.combinations[k],
                )
                self.__validate_key_for_type(dc_type, k)
        self.__check_for_missing_keys(dc_type, dc_keys)
