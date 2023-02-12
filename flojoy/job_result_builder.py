from enum import Enum
from flojoy import flojoy, DataContainer
import traceback

from flojoy.flojoy_instruction import FLOJOY_INSTRUCTION


class JobResultBuilder:

    def __init__(self) -> None:
        self.data = self.instructions = None

    def _add_instruction(self, instruction):
        self.instructions = self.instructions if self.instructions is not None else {}
        self.instructions = {
            **self.instructions,
            **instruction,
        }
    
    def from_params(self, params):
        try:
            x = params[0].x
            y = params[0].y
        except Exception:
            print(traceback.format_exc())

        self.data = DataContainer(x = x, y = y)

        return self

    def from_data(self, data):
        self.data = data
        return self

    def flow_to_nodes(self, nodes):
        self._add_instruction({
            FLOJOY_INSTRUCTION.FLOW_TO_NODES: nodes
        })
        return self

    def flow_to_directions(self, directions):
        self._add_instruction({
            FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS: directions
        })
        return self

    def build(self):
        result = self.data
        if self.instructions:
            result = {
                **self.instructions,
                # check fetch_input method in flojoy.py to see how this is processedÒ
                FLOJOY_INSTRUCTION.RESULT_FIELD: "data",
                'data': result,
            }
        return result
