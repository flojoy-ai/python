from flojoy.flojoy_instruction import FLOJOY_INSTRUCTION
import traceback


def is_flow_controled(result):
    if FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS in result or FLOJOY_INSTRUCTION.FLOW_TO_NODES in result:
        return True


def get_next_directions(result):
    return result.get(FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS, [])


def get_next_nodes(result):
    return result.get(FLOJOY_INSTRUCTION.FLOW_TO_NODES, [])


def get_data(result):
    if "__result__field__" in result:
        data = result[result["__result__field__"]]
    else:
        data = result

    return data
