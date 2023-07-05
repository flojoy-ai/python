import os
import json
import yaml
import traceback
from rq.job import Job  # type:ignore
from pathlib import Path
from functools import wraps
from .data_container import DataContainer
from .utils import PlotlyJSONEncoder, dump_str
from typing import Callable, Any, Optional
from .job_result_utils import get_frontend_res_obj_from_result, get_dc_from_result
from .utils import redis_instance, send_to_socket
from .parameter_types import format_param_value
from time import sleep
from inspect import signature

__all__ = ["flojoy", "DefaultParams"]


def get_flojoy_root_dir() -> str:
    home = str(Path.home())
    path = os.path.join(home, ".flojoy/flojoy.yaml")
    stream = open(path, "r")
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    root_dir = ""
    if isinstance(yaml_dict, str):
        root_dir = yaml_dict.split(":")[1]
    else:
        root_dir = yaml_dict["PATH"]
    return root_dir


def fetch_inputs(previous_jobs: list[dict[str, str]]):
    """
    Queries Redis for job results

    Parameters
    ----------
    previous_jobs : list of Redis jobs that directly precede this node.
    Each item representing a job contains `job_id` and `input_name`.
    `input_name` is the port where the previous job with `job_id` connects to.

    Returns
    -------
    inputs : list of DataContainer objects
    """
    dict_inputs: dict[str, DataContainer | list[DataContainer]] = dict()

    try:
        for prev_job in previous_jobs:
            num_of_time_attempted = 0
            prev_job_id = prev_job.get("job_id")
            input_name = prev_job.get("input_name", "")
            multiple = prev_job.get("multiple", False)
            edge = prev_job.get("edge", "")
            print(
                "fetching input from prev job id:",
                prev_job_id,
                " for input:",
                input_name,
                "edge: ",
                edge,
            )
            while num_of_time_attempted < 3:
                job = Job.fetch(prev_job_id, connection=redis_instance)  # type:ignore
                job_result: dict[str, Any] = job.result  # type:ignore
                result = (
                    get_dc_from_result(job_result[edge])
                    if edge != "default"
                    else get_dc_from_result(job_result)
                )
                print(
                    "fetch input from prev job id:",
                    prev_job_id,
                    " result:",
                    dump_str(result, limit=100),
                )
                if result is not None:
                    if multiple:
                        if input_name not in dict_inputs:
                            dict_inputs[input_name] = [result]
                        else:
                            dict_inputs[input_name].append(result)
                    else:
                        dict_inputs[input_name] = result
                    break
                else:
                    sleep(0.3)
                    num_of_time_attempted += 1

    except Exception:
        print(traceback.format_exc())
    return dict_inputs


def get_redis_obj(id: str) -> dict[str, Any]:
    get_obj = redis_instance.get(id)
    parse_obj: dict[str, Any] = json.loads(get_obj) if get_obj is not None else {}
    return parse_obj


class DefaultParams:
    def __init__(
        self, node_id: str, job_id: str, jobset_id: str, node_type: str
    ) -> None:
        self.node_id = node_id
        self.job_id = job_id
        self.jobset_id = jobset_id
        self.node_type = node_type


def flojoy(
    original_function: Callable[..., DataContainer | dict[str, Any]] | None = None,
    *,
    node_type: Optional[str] = None,
    deps: Optional[dict[str, str]] = None,
    inject_node_metadata: bool = False,
):
    """
    Decorator to turn Python functions with numerical return
    values into Flojoy nodes.

    @flojoy is intended to eliminate boilerplate in connecting
    Python scripts as visual nodes

    Into whatever function it wraps, @flojoy injects
    1. the last node's input as list of DataContainer object
    2. parameters for that function (either set by the user or default)

    Parameters
    ----------
    `func`: Python function that returns DataContainer object

    Returns
    -------
    A dict containing DataContainer object

    Usage Example
    -------------
    ```
    @flojoy
    def SINE(dc_inputs:list[DataContainer], params:dict[str, Any]):

        print('params passed to SINE', params)

        dc_input = dc_inputs[0]

        output = DataContainer(
            x=dc_input.x,
            y=np.sin(dc_input.x)
        )
        return output
    ```

    ## equivalent to: decorated_sine = flojoy(SINE)
    ```
    pj_ids = [123, 456]
    print(SINE(previous_job_ids = pj_ids, mock = True))
    ```
    """

    def decorator(func: Callable[..., DataContainer | dict[str, Any]]):
        @wraps(func)
        def wrapper(
            node_id: str,
            job_id: str,
            jobset_id: str,
            previous_jobs: list[dict[str, str]] = [],
            ctrls: dict[str, Any] | None = None,
        ):
            try:
                FN = func.__name__
                # remove this node from redis ALL_NODES key
                redis_instance.lrem(f"{jobset_id}_ALL_NODES", 1, job_id)
                sys_status = "🏃‍♀️ Running python job: " + FN
                send_to_socket(
                    json.dumps(
                        {
                            "SYSTEM_STATUS": sys_status,
                            "jobsetId": jobset_id,
                            "RUNNING_NODE": node_id,
                        }
                    )
                )
                print("previous jobs:", previous_jobs)
                # Get command parameters set by the user through the control panel
                func_params: dict[str, Any] = {}
                if ctrls is not None:
                    for _, input in ctrls.items():
                        param = input["param"]
                        value = input["value"]
                        func_params[param] = format_param_value(value, input["type"])
                func_params["type"] = "default"

                print("executing node_id:", node_id, "previous_jobs:", previous_jobs)
                dict_inputs = fetch_inputs(previous_jobs)

                # constructing the inputs
                print(f"constructing inputs for {func.__name__}")
                args: dict[str, Any] = {}
                sig = signature(func)

                args = {**args, **dict_inputs}

                for param, value in func_params.items():
                    if param in sig.parameters:
                        args[param] = value
                if inject_node_metadata:
                    args["default_params"] = DefaultParams(
                        job_id=job_id,
                        node_id=node_id,
                        jobset_id=jobset_id,
                        node_type="default",
                    )

                print(node_id, " params: ", args.keys())

                ##########################
                # calling the node function
                ##########################
                dc_obj = func(**args)  # DataContainer object from node
                ##########################
                # end calling the node function
                ##########################

                # some special nodes like LOOP return dict instead of `DataContainer`
                if isinstance(dc_obj, DataContainer):
                    dc_obj.validate()  # Validate returned DataContainer object
                else:
                    for value in dc_obj.values():
                        if isinstance(value, DataContainer):
                            value.validate()
                # Response object to send to FE
                result = get_frontend_res_obj_from_result(dc_obj)

                send_to_socket(
                    json.dumps(
                        {
                            "NODE_RESULTS": {
                                "cmd": FN,
                                "id": node_id,
                                "result": result,
                            },
                            "jobsetId": jobset_id,
                        },
                        cls=PlotlyJSONEncoder,
                    )
                )

                if func.__name__ == "END":
                    send_to_socket(
                        json.dumps(
                            {
                                "SYSTEM_STATUS": "🤙 python script run successful",
                                "RUNNING_NODE": "",
                                "jobsetId": jobset_id,
                            }
                        )
                    )
                print("final result:", dump_str(result, limit=100))
                return dc_obj
            except Exception as e:
                send_to_socket(
                    json.dumps(
                        {
                            "SYSTEM_STATUS": f"Failed to run: {func.__name__}",
                            "FAILED_NODES": node_id,
                            "FAILURE_REASON": e.args[0],
                            "jobsetId": jobset_id,
                        }
                    )
                )
                print("error occured while running the node")
                print(traceback.format_exc())
                raise e

        return wrapper

    if original_function:
        return decorator(original_function)

    return decorator
