import plotly.express as px
import numpy as np

dtypes = ['grayscale', 'matrix', 'dataframe',
          'image', 'ordered_pair', 'ordered_triple', 'scalar']


def data_container_to_plotly(data:dict, plot_type=None):
    value = data.copy()
    d_type = value.type
    fig = {}
    if 'x' in value and isinstance(value.x, dict):
        # value.x = []
        # for k, v in data.x.items():
        #     value.x.append(v)
        dict_keys = list(value.x.keys())
        value.x = value.x[dict_keys[0]]
        
    match d_type:
        case 'image':
            if value.a is None:
                img_combined = np.stack((value.r, value.g, value.b), axis=2)
            else:
                img_combined = np.stack(
                    (value.r, value.g, value.b, value.a), axis=2)
            fig = px.imshow(img=img_combined)
        case 'ordered_pair':
            if plot_type:
                fig = get_plot_fig_by_type(
                    plot_type=plot_type, x=value.x, y=value.y)
            else:
                fig = px.line(x=value.x, y=value.y)
        case 'ordered_triple':
            if plot_type:
                fig = get_plot_fig_by_type(
                    plot_type=plot_type, x=value.x, y=value.y, z=value.z)
            else:
                fig = px.scatter_3d(x=value.x, y=value.y, z=value.z)
        case 'scalar':
            if plot_type:
                fig = get_plot_fig_by_type(
                    plot_type=plot_type, x=value.c)  # x or y ?
            else:
                fig = px.histogram(x=value.c)
        case 'grayscale' | 'matrix' | 'dataframe':
            if plot_type:
                fig = get_plot_fig_by_type(plot_type=plot_type, x=value.m)
            else:
                fig = px.histogram(x=value.m)
        case 'plotly':
            fig = data.f
        case _:
            raise ValueError('Not supported DataContainer type!')
    return fig.to_dict()


def get_plot_fig_by_type(plot_type, **kwargs):
    match plot_type:
        case 'bar':
            fig = px.bar(**kwargs)
        case 'histogram':
            fig = px.histogram(**kwargs)
        case 'scatter':
            fig = px.scatter(**kwargs)
        case 'scatter_3d':
            fig = px.scatter_3d(**kwargs)
        case 'image':
            fig = px.imshow(**kwargs)
        case _:
            fig = px.line(**kwargs)
    return fig
