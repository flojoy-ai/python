from .flojoy_instruction import FLOJOY_INSTRUCTION
from .plotly_utils import data_container_to_plotly
from .data_container import DataContainer
from .dao import Dao
from typing import Any, cast

__all__ = ["get_job_result", "get_next_directions", "get_next_nodes", "get_job_result"]


def is_flow_controled(result: dict[str, Any] | DataContainer):
    if (
        FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS in result
        or FLOJOY_INSTRUCTION.FLOW_TO_NODES in result
    ):
        return True
    return False


def get_next_directions(result: dict[str, Any] | None) -> list[str] | None:
    direction = None
    if result is None:
        return direction
    if not result.get(FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS):
        for value in result.values():
            if isinstance(value, dict) and value.get(
                FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS
            ):
                direction = cast(
                    list[str], value[FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS]
                )
                break
    else:
        direction = result[FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS]
    return direction


def get_next_nodes(result: dict[str, Any] | None) -> list[str]:
    if result is None:
        return []
    return cast(list[str], result.get(FLOJOY_INSTRUCTION.FLOW_TO_NODES, []))


def get_dc_from_result(
    result: dict[str, Any] | DataContainer | None
) -> DataContainer | None:
    if not result:
        return None
    if isinstance(result, DataContainer):
        return result
    if result.get(FLOJOY_INSTRUCTION.RESULT_FIELD):
        return result[result[FLOJOY_INSTRUCTION.RESULT_FIELD]]
    return result["data"]


def get_job_result(job_id: str) -> dict[str, Any] | DataContainer | None:
    try:
        job_result: Any = Dao.get_instance().get_job_result(job_id)
        result = get_dc_from_result(cast(dict[str, Any] | DataContainer, job_result))
        return result
    except Exception:
        return None


def get_frontend_res_obj_from_result(
    result: dict[str, Any] | DataContainer
) -> dict[str, Any]:
    if not result:
        return {"default_fig": result, "data": result}
    if isinstance(result, DataContainer):
        plotly_fig = data_container_to_plotly(data=result)
        return {"default_fig": plotly_fig, "data": result}
    if result.get(FLOJOY_INSTRUCTION.RESULT_FIELD):
        data = result[result[FLOJOY_INSTRUCTION.RESULT_FIELD]]
        plotly_fig = data_container_to_plotly(data=data)
        return {**result, "default_fig": plotly_fig, "data": data}
    keys = list(result.keys())
    return get_frontend_res_obj_from_result(result[keys[0]])
