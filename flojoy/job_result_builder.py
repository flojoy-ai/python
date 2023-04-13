from flojoy import DataContainer
import numpy as np
from typing import TypedDict
from flojoy.flojoy_instruction import FLOJOY_INSTRUCTION
from flojoy.plotly_utils import data_container_to_plotly
from plotly.graph_objects import Figure


class JobResultBuilder:


    def __init__(self) -> None:
        self.instructions = None
        self.data = self.get_default_data()

    def _add_instructions(self, instruction):
        self.instructions = self.instructions if self.instructions is not None else {}
        self.instructions = {
            **self.instructions,
            **instruction,
        }

    def from_inputs(self, inputs: list[DataContainer]):
        # if no inputs were provided, construct fake output
        if len(inputs) == 0:
            self.data = self.get_default_data()
        else:
            self.data = inputs[0]

        return self

    def from_data(self, data):
        self.data = data
        return self

    def flow_to_nodes(self, nodes):
        if not nodes:
            return self
        self._add_instructions({
            FLOJOY_INSTRUCTION.FLOW_TO_NODES: nodes
        })
        return self

    def flow_to_directions(self, directions):
        if not directions:
            return self
        self._add_instructions({
            FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS: directions
        })
        return self

    def flow_by_flag(self, flag, directionsWhenTrue, directionsWhenFalse):
        self._add_instructions({
            FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS: directionsWhenTrue if flag else directionsWhenFalse
        })
        return self

    def build(self):
        result = self.data
        if self.instructions:
            result = {
                # instructions to job scheduler (watch.py)
                **self.instructions,
                # instruction to fetch_input method in flojoy.py
                FLOJOY_INSTRUCTION.RESULT_FIELD: "data",
                'data': result,
            }
        return result
    class To_plot_args(TypedDict):
        plot_type: str
        fig: Figure
        data_container_plotly: DataContainer
        
    def to_plot(self, args: To_plot_args):
        if not self.instructions:
            self.instructions = {}
        if args.get('fig', None) is not None:
            fig = args['fig'] # get_plot_fig_by_type(plot_type=plot_type, **kwargs)
            output = fig.to_dict()
        elif args.get('data_container_plotly',None) is not None:
            output = data_container_to_plotly(args['data_container_plotly'], None)
        else:
            output = data_container_to_plotly(self.data, plot_type=args['plot_type'])

        return {
            **self.instructions,
            FLOJOY_INSTRUCTION.DATACONTAINER_FIELD: 'result',
            'result': self.data,
            FLOJOY_INSTRUCTION.RESULT_FIELD: 'data',
            'data': output
        }
    def get_default_data(self):
        x = np.arange(0,1000,1)
        y = np.ones_like(x)
        return DataContainer(x=x, y=y)
