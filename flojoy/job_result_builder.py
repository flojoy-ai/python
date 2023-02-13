from flojoy import DataContainer
import traceback
import numpy as np

from flojoy.flojoy_instruction import FLOJOY_INSTRUCTION


class JobResultBuilder:

    def __init__(self) -> None:
        self.data = self.instructions = None

    def _add_instructions(self, instruction):
        self.instructions = self.instructions if self.instructions is not None else {}
        self.instructions = {
            **self.instructions,
            **instruction,
        }
    
    def from_inputs(self, inputs):
        # if no inputs were provided, construct fake output
        if len(inputs) == 0 or np.any(inputs[0].y) == None:
            x = list()
            for i in range(1000):
                x.append(i)
            y = np.full(1000, 1)

            self.data = DataContainer(x=x, y=y)
        else:
            self.data = inputs[0]

        return self

    def from_data(self, data):
        self.data = data
        return self

    def flow_to_nodes(self, nodes):
        self._add_instructions({
            FLOJOY_INSTRUCTION.FLOW_TO_NODES: nodes
        })
        return self

    def flow_to_directions(self, directions):
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
                **self.instructions, # instructions to job scheduler (watch.py)                
                FLOJOY_INSTRUCTION.RESULT_FIELD: "data", # instruction to fetch_input method in flojoy.py
                'data': result,
            }
        return result
