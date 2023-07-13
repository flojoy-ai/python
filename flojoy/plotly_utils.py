import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from .data_container import DataContainer
import pandas as pd
from typing import cast, Any


def data_container_to_plotly(data: DataContainer) -> dict[str, Any]:
    data_copy = data.copy()
    dc_type = data_copy.type
    fig = go.Figure()
    if "x" in data_copy and isinstance(data_copy.x, dict):  # type: ignore
        data_keys = list(cast(list[str], data_copy.x.keys()))
        data_copy.x = data_copy.x[data_keys[0]]  # type: ignore

    match dc_type:
        case "image":
            if data_copy.a is None:
                img_combined = np.stack((data_copy.r, data_copy.g, data_copy.b), axis=2)
            else:
                img_combined = np.stack(
                    (data_copy.r, data_copy.g, data_copy.b, data_copy.a), axis=2
                )
            fig = px.imshow(img=img_combined)  # type:ignore
        case "ordered_pair":
            if data_copy.x is not None and len(data_copy.x) != len(data_copy.y):
                data_copy.x = np.arange(0, len(data_copy.y), 1)
            fig = px.line(x=data_copy.x, y=data_copy.y)
        case "ordered_triple":
            fig = px.scatter_3d(x=data_copy.x, y=data_copy.y, z=data_copy.z)
        case "scalar":
            fig = px.histogram(x=data_copy.c)
        case "dataframe":
            df = cast(pd.DataFrame, data_copy.m)
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(df.columns)),
                        cells=dict(values=[df[col] for col in df.columns]),
                    )
                ]
            )
        case "grayscale" | "matrix":
            y_columns: np.ndarray = data_copy.m
            for i, col in enumerate(y_columns.T):
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(0, col.size),
                        y=col,
                        mode="lines",
                        name=i,
                    )
                )
        case "plotly":
            fig = cast(go.Figure, data.fig)
        case _:
            raise ValueError(
                f"unsupported DataContainer type passed to plotly converter function, type: '{dc_type}"
            )
    return cast(dict[str, Any], fig.to_dict())
