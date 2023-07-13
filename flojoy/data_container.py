import typing
import numpy as np
import pandas as pd
from box import Box, box_list
import plotly.graph_objects as go  # type:ignore
from typing import Union, Literal, get_args, Any, cast
from .utils import find_closest_match


DCType = Literal[
    "scalar",
    "vector",
    "matrix",
    "dataframe",
    "grayscale",
    "image",
    "ordered_pair",
    "ordered_triple",
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
ExtraType = dict[str, Any] | None


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

    allowed_types = list(typing.get_args(DCType))
    allowed_keys = [
        "x",
        "y",
        "z",
        "t",
        "v",
        "m",
        "c",
        "r",
        "g",
        "b",
        "a",
        "fig",
        "extra",
    ]
    combinations = {
        "x": ["y", "t", "z", "fig", "extra"],
        "y": ["x", "t", "z", "fig", "extra"],
        "z": ["x", "y", "t", "fig", "extra"],
        "c": ["t", "fig", "extra"],
        "v": ["t", "fig", "extra"],
        "m": ["t", "fig", "extra"],
        "t": [*(value for value in allowed_keys if value not in ["t"])],
        "r": ["g", "b", "t", "a", "fig", "extra"],
        "g": ["r", "b", "t", "a", "fig", "extra"],
        "b": ["r", "g", "t", "a", "fig", "extra"],
        "a": ["r", "g", "b", "t", "fig", "extra"],
        "extra": [*(k for k in allowed_keys if k not in ["extra"])],
        "fig": [*(k for k in allowed_keys if k not in ["fig"])],
    }
    type_keys_map: dict[DCType, list[str]] = {
        "dataframe": ["m"],
        "vector": ["v"],
        "matrix": ["m"],
        "grayscale": ["m"],
        "image": ["r", "g", "b", "a"],
        "ordered_pair": ["x", "y"],
        "ordered_triple": ["x", "y", "z"],
        "scalar": ["c"],
        "plotly": ["fig"],
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
                f"DataContainer keys must be any of "
                f"following types: {get_args(DCKwargsValue)}"
            )

    def __init__(  # type:ignore
        self, type: DCType = "ordered_pair", **kwargs: DCKwargsValue
    ):
        self.type = type
        for k, v in kwargs.items():
            self[k] = v

    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)

    def __getitem__(self, key: str, _ignore_default: bool = False) -> Any:
        return super().__getitem__(key, _ignore_default)  # type:ignore

    def __setitem__(self, key: str, value: DCKwargsValue) -> None:
        if key != "type" and key != "extra":
            formatted_value = self._ndarrayify(value)
            super().__setitem__(key, formatted_value)  # type:ignore
        else:
            super().__setitem__(key, value)  # type: ignore

    def __check_combination(self, key: str, keys: list[str], allowed_keys: list[str]):
        for i in keys:
            if i not in allowed_keys:
                raise ValueError(f"You cant have {key} with {i}")

    def __validate_key_for_type(self, data_type: DCType, key: str):
        if data_type.startswith("parametric_") and key != "t":
            splitted_type = cast(DCType, data_type.split("parametric_")[1])
            self.__validate_key_for_type(splitted_type, key)
        else:
            if (
                key not in self.type_keys_map[data_type] + ["extra"]
                and data_type != "plotly"
            ):
                raise KeyError(
                    self.__build_error_text(
                        key, data_type, self.type_keys_map[data_type]
                    )
                )

    def __check_for_missing_keys(self, dc_type: DCType, keys: list[str]):
        if dc_type.startswith("parametric_"):
            if "t" not in keys:
                raise KeyError(f't key must be provided for "{dc_type}"')
            t = self["t"]
            is_ascending_order = all(t[i] <= t[i + 1] for i in range(len(t) - 1))
            if is_ascending_order is not True:
                raise ValueError("t key must be in ascending order")
            splitted_type = cast(DCType, dc_type.split("parametric_")[1])
            self.__check_for_missing_keys(splitted_type, keys)
        else:
            for k in self.type_keys_map[dc_type]:
                if k not in keys:
                    raise KeyError(f'"{k}" key must be provided for type "{dc_type}"')

    def __build_error_text(self, key: str, data_type: str, available_keys: list[str]):
        return (
            f'Invalid key "{key}" provided for data type "{data_type}", '
            f'supported keys: {", ".join(available_keys)}'
        )

    def validate(self):
        dc_type = self.type
        if dc_type not in self.allowed_types:
            closest_type = find_closest_match(dc_type, self.allowed_types)
            helper_text = (
                f'Did you mean: "{closest_type}" ?'
                if closest_type
                else f'allowed types: "{", ".join(self.allowed_types)}"'
            )
            raise ValueError(
                f'unsupported type "{dc_type}" passed to '
                f"DataContainer class, {helper_text}"
            )
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


class OrderedPair(DataContainer):
    x: DCNpArrayType
    y: DCNpArrayType

    def __init__(
        self, x: DCNpArrayType, y: DCNpArrayType, extra: ExtraType = None
    ):  # type:ignore
        super().__init__(type="ordered_pair", x=x, y=y, extra=extra)


class ParametricOrderedPair(DataContainer):
    x: DCNpArrayType
    y: DCNpArrayType
    t: DCNpArrayType

    def __init__(
        self,
        x: DCNpArrayType,
        y: DCNpArrayType,
        t: DCNpArrayType,
        extra: ExtraType = None,
    ):  # type:ignore
        super().__init__(type="parametric_ordered_pair", x=x, y=y, t=t, extra=extra)


class OrderedTriple(DataContainer):
    x: DCNpArrayType
    y: DCNpArrayType
    z: DCNpArrayType

    def __init__(
        self,
        x: DCNpArrayType,
        y: DCNpArrayType,
        z: DCNpArrayType,
        extra: ExtraType = None,
    ):  # type:ignore
        super().__init__(type="ordered_triple", x=x, y=y, z=z, extra=extra)


class ParametricOrderedTriple(DataContainer):
    x: DCNpArrayType
    y: DCNpArrayType
    z: DCNpArrayType
    t: DCNpArrayType

    def __init__(  # type:ignore
        self,
        x: DCNpArrayType,
        y: DCNpArrayType,
        z: DCNpArrayType,
        t: DCNpArrayType,
        extra: ExtraType = None,
    ):
        super().__init__(
            type="parametric_ordered_triple", x=x, y=y, z=z, t=t, extra=extra
        )


class Scalar(DataContainer):
    c: int | float

    def __init__(self, c: int | float, extra: ExtraType = None):  # type:ignore
        super().__init__(type="scalar", c=c, extra=extra)


class ParametricScalar(DataContainer):
    c: int | float
    t: DCNpArrayType

    def __init__(
        self, c: int | float, t: DCNpArrayType, extra: ExtraType = None
    ):  # type:ignore
        super().__init__(type="scalar", c=c, t=t, extra=extra)


class Vector(DataContainer):
    v: DCNpArrayType

    def __init__(self, v: DCNpArrayType, extra: ExtraType = None):  # type:ignore
        super().__init__(type="vector", v=v, extra=extra)


class ParametricVector(DataContainer):
    v: DCNpArrayType

    def __init__(
        self, v: DCNpArrayType, t: DCNpArrayType, extra: ExtraType = None
    ):  # type:ignore
        super().__init__(type="vector", v=v, t=t, extra=extra)


class Matrix(DataContainer):
    m: DCNpArrayType

    def __init__(self, m: DCNpArrayType, extra: ExtraType = None):  # type:ignore
        super().__init__(type="matrix", m=m, extra=extra)


class ParametricMatrix(DataContainer):
    m: DCNpArrayType
    t: DCNpArrayType

    def __init__(
        self, m: DCNpArrayType, t: DCNpArrayType, extra: ExtraType = None
    ):  # type:ignore
        super().__init__(type="matrix", m=m, t=t, extra=extra)


class DataFrame(DataContainer):
    m: pd.DataFrame

    def __init__(self, df: pd.DataFrame, extra: ExtraType = None):
        super().__init__(type="dataframe", m=df, extra=extra)


class ParametricDataFrame(DataContainer):
    m: pd.DataFrame
    t: DCNpArrayType

    def __init__(
        self, df: pd.DataFrame, t: DCNpArrayType, extra: ExtraType = None
    ):  # type:ignore
        super().__init__(type="dataframe", m=df, t=t, extra=extra)


class Plotly(DataContainer):
    fig: go.Figure

    def __init__(self, fig: go.Figure, extra: ExtraType = None):  # type:ignore
        super().__init__(type="plotly", fig=fig, extra=extra)


class ParametricPlotly(DataContainer):
    fig: go.Figure
    t: DCNpArrayType

    def __init__(
        self, fig: go.Figure, t: DCNpArrayType, extra: ExtraType = None
    ):  # type:ignore
        super().__init__(type="plotly", fig=fig, t=t, extra=extra)


class Image(DataContainer):
    r: DCNpArrayType
    g: DCNpArrayType
    b: DCNpArrayType
    a: DCNpArrayType | None

    def __init__(  # type:ignore
        self,
        r: DCNpArrayType,
        g: DCNpArrayType,
        b: DCNpArrayType,
        a: DCNpArrayType | None = None,
        extra: ExtraType = None,
    ):
        super().__init__(type="image", r=r, g=g, b=b, a=a, extra=extra)


class ParametricImage(DataContainer):
    t: DCNpArrayType
    r: DCNpArrayType
    g: DCNpArrayType
    b: DCNpArrayType
    a: DCNpArrayType | None

    def __init__(  # type:ignore
        self,
        r: DCNpArrayType,
        g: DCNpArrayType,
        b: DCNpArrayType,
        a: DCNpArrayType,
        t: DCNpArrayType,
        extra: ExtraType = None,
    ):
        super().__init__(type="image", r=r, g=g, b=b, a=a, t=t, extra=extra)


class Grayscale(DataContainer):
    m: DCNpArrayType

    def __init__(self, img: DCNpArrayType, extra: ExtraType = None):  # type:ignore
        super().__init__(type="grayscale", m=img, extra=extra)


class ParametricGrayscale(DataContainer):
    m: DCNpArrayType
    t: DCNpArrayType

    def __init__(
        self, img: DCNpArrayType, t: DCNpArrayType, extra: ExtraType = None
    ):  # type:ignore
        super().__init__(type="grayscale", m=img, t=t, extra=extra)
