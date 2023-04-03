import plotly.express as px
import numpy as np

dtypes = ['grayscale', 'matrix', 'dataframe',
          'image', 'ordered_pair', 'ordered_triple', 'scalar']


def data_container_to_plotly(data, plot_type):
    d_type = data.type
    fig = {}
    if 'x' in data and isinstance(data.x, dict):
        data.x = data.x.a
        
    match d_type:
        case 'image':
            if data.a is None:
                img_combined = np.stack((data.r, data.g, data.b), axis=2)
            else:
                img_combined = np.stack(
                    (data.r, data.g, data.b, data.a), axis=2)
            fig = px.imshow(img=img_combined)
        case 'ordered_pair':
            if plot_type:
                fig = get_plot_fig_by_type(
                    plot_type=plot_type, x=data.x, y=data.y)
            else:
                fig = px.line(x=data.x, y=data.y)
        case 'ordered_triple':
            if plot_type:
                fig = get_plot_fig_by_type(
                    plot_type=plot_type, x=data.x, y=data.y, z=data.z)
            else:
                fig = px.scatter_3d(x=data.x, y=data.y, z=data.z)
        case 'scalar':
            if plot_type:
                fig = get_plot_fig_by_type(
                    plot_type=plot_type, x=data.c)  # x or y ?
            else:
                fig = px.histogram(x=data.c)
        case 'grayscale' | 'matrix' | 'dataframe':
            if plot_type:
                fig = get_plot_fig_by_type(plot_type=plot_type, x=data.m)
            else:
                fig = px.histogram(x=data.m)
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
