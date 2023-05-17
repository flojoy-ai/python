import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def data_container_to_plotly(data: dict):
    value = data.copy()
    d_type = value.type
    fig = {}
    if "x" in value and isinstance(value.x, dict):
        # value.x = []
        # for k, v in data.x.items():
        #     value.x.append(v)
        dict_keys = list(value.x.keys())
        value.x = value.x[dict_keys[0]]

    match d_type:
        case "image":
            if value.a is None:
                img_combined = np.stack((value.r, value.g, value.b), axis=2)
            else:
                img_combined = np.stack((value.r, value.g, value.b, value.a), axis=2)
            fig = px.imshow(img=img_combined)
        case "ordered_pair":
            if value.x is not None and len(value.x) != len(value.y):
                value.x = np.arange(0, len(value.y), 1)
            fig = px.line(x=value.x, y=value.y)
        case "ordered_triple":
            fig = px.scatter_3d(x=value.x, y=value.y, z=value.z)
        case "scalar":
            fig = px.histogram(x=value.c)
        case "dataframe":
            df = value.m
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(df.columns)),
                        cells=dict(values=[df[col] for col in df.columns]),
                    )
                ]
            )
        case "grayscale" | "matrix":
            fig = px.histogram(x=value.m)
        case "plotly":
            fig = data.fig
        case _:
            raise ValueError("Not supported DataContainer type!")
    return fig.to_dict()
