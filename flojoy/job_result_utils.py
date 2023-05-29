from flojoy.flojoy_instruction import FLOJOY_INSTRUCTION
from flojoy.plotly_utils import data_container_to_plotly
from rq.job import Job  # type:ignore
from .utils import redis_instance
from .data_container import DataContainer
from typing import Any, cast

__all__ = ["get_job_result"]


def is_flow_controled(result: dict[str, Any] | DataContainer):
    if (
        FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS in result
        or FLOJOY_INSTRUCTION.FLOW_TO_NODES in result
    ):
        return True
    return False


def get_next_directions(result: dict[str, Any] | None) -> list[str]:
    if result is None:
        return ["main"]
    return result.get(FLOJOY_INSTRUCTION.FLOW_TO_DIRECTIONS, ["main"])


def get_next_nodes(result: dict[str, Any] | None) -> list[str]:
    if result is None:
        return []
    return cast(list[str], result.get(FLOJOY_INSTRUCTION.FLOW_TO_NODES, []))


def get_dc_from_result(result: dict[str, Any] | DataContainer) -> DataContainer | None:
    if not result:
        return None
    if isinstance(result, DataContainer):
        return result
    if result.get(FLOJOY_INSTRUCTION.RESULT_FIELD):
        return result[result[FLOJOY_INSTRUCTION.RESULT_FIELD]]
    return result["data"]


def get_job_result(job_id: str) -> DataContainer | None:
    job = Job.fetch(job_id, connection=redis_instance)  # type:ignore
    job_result: dict[str, Any] = job.result  # type:ignore
    result = get_dc_from_result(cast(dict[str, Any] | DataContainer, job_result))
    return result


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
    return result
