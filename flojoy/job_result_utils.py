from flojoy.flojoy_instruction import FLOJOY_INSTRUCTION


def is_flow_controled(result):
    if FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS in result or FLOJOY_INSTRUCTION.FLOW_TO_NODES in result:
        return True


def get_next_directions(result):
    if result is None: return []
    return result.get(FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS, [])


def get_next_nodes(result):
    if result is None: return []
    return result.get(FLOJOY_INSTRUCTION.FLOW_TO_NODES, [])


def get_data(result):
    if result and result.get(FLOJOY_INSTRUCTION.RESULT_FIELD):
        data = result[result[FLOJOY_INSTRUCTION.RESULT_FIELD]]
    else:
        data = result

    return data
