import os, json, yaml, traceback
import numpy as np
import networkx as nx
from pathlib import Path
from functools import wraps
from .data_container import DataContainer
from .utils import PlotlyJSONEncoder, dump_str
from networkx.drawing.nx_pylab import draw as nx_draw  # type: ignore
from typing import Union, cast, Any, Literal, Callable, List, Optional, Type
from .job_result_utils import get_frontend_res_obj_from_result, get_dc_from_result
from .utils import send_to_socket
from time import sleep
from inspect import signature
from .job_service import JobService

def get_flojoy_root_dir() -> str:
    home = str(Path.home())
    path = os.path.join(home, ".flojoy/flojoy.yaml")
    stream = open(path, "r")
    yaml_dict = yaml.load(stream, Loader=yaml.FullLoader)
    root_dir = ""
    if isinstance(yaml_dict, str) == True:
        root_dir = yaml_dict.split(":")[1]
    else:
        root_dir = yaml_dict["PATH"]
    return root_dir


def fetch_inputs(
    previous_jobs: list[dict[str, str]]
):
    """
    Queries for job results

    Parameters
    ----------
    previous_jobs : list of jobs that directly precede this node.
    Each item representing a job contains `job_id` and `input_name`.
    `input_name` is the port where the previous job with `job_id` connects to.

    Returns
    -------
    inputs : list of DataContainer objects
    """
    inputs: list[DataContainer] = []
    dict_inputs: dict[str, DataContainer] = dict()

    try:


        for prev_job in previous_jobs:
            num_of_time_attempted = 0
            prev_job_id = prev_job.get("job_id")
            input_name = prev_job.get("input_name","")
            print(
                "fetching input from prev job id:",
                prev_job_id,
                " for input:",
                input_name,
                flush=True
            )
            while num_of_time_attempted < 3:
                job_result = JobService().get_job_result(prev_job_id)  
                result = get_dc_from_result(job_result)
                if result is not None:
                    print(f"got job result from {prev_job_id}", flush=True)
                    inputs.append(result)
                    dict_inputs[input_name] = result
                    break
                else:
                    print(f"didn't get job result from {prev_job_id}", flush=True)
                    sleep(0.05)
                    num_of_time_attempted += 1
    except Exception:
        print(traceback.format_exc(), flush=True)

    return inputs, dict_inputs


def parse_array(str_value: str) -> List[Union[int, float, str]]:
    if not str_value:
        return []

    val_list = [val.strip() for val in str_value.split(",")]
    val = list(map(str, val_list))
    # First try to cast into int, then float, then keep as string if all else fails
    for t in [int, float]:
        try:
            val: list[int | float | str] = list(map(t, val_list))
            break
        except Exception:
            continue
    return val


ParamValTypes = Literal[
    "array", "float", "int", "boolean", "select", "node_reference", "string"
]


def format_param_value(value: Any, value_type: ParamValTypes):
    match value_type:
        case "array":
            s = str(value)
            parsed_value = parse_array(s)
            return parsed_value
        case "float":
            return float(value)
        case "int":
            return int(value)
        case "boolean":
            return bool(value)
        case "select" | "string" | "node_reference":
            return str(value)
    return value


flojoyKwargs = Union[str, dict[str, dict[str, str]], list[str]]


def flojoy(func: Callable[..., DataContainer | dict[str, Any]]):
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

    @wraps(func)
    def wrapper(**kwargs: flojoyKwargs):
        node_id = cast(str, kwargs["node_id"])
        job_id = cast(str, kwargs["job_id"])
        jobset_id = cast(str, kwargs["jobset_id"])
        try:
            previous_jobs = cast(list[dict[str, str]], kwargs.get("previous_jobs", []))
            ctrls = cast(
                Union[dict[str, dict[str, Any]], None], kwargs.get("ctrls", None)
            )
            FN = func.__name__
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
            # Get command parameters set by the user through the control panel
            func_params = {}
            if ctrls is not None:
                for _, input in ctrls.items():
                    param = input["param"]
                    value = input["value"]
                    func_params[param] = format_param_value(
                        value,
                        input["type"]
                        if "type" in input
                        else type(
                            value
                        ),  # else condition is for backward compatibility
                    )
            func_params["jobset_id"] = jobset_id
            func_params["type"] = "default"
            func_params["node_id"] = node_id
            func_params["job_id"] = job_id

            print("executing node_id:", node_id, "previous_jobs:", previous_jobs, flush=True)
            print(node_id, " params: ", json.dumps(func_params, indent=2), flush=True)
            node_inputs, dict_inputs = fetch_inputs(previous_jobs)

            # constructing the inputs
            print(f"constructing inputs for {func}", flush=True)
            args = {}
            sig = signature(func)

            # once all the nodes are migrated to the new node api, remove the if condition
            keys = list(sig.parameters)
            if (
                len(sig.parameters) == 2
                and sig.parameters[keys[0]].annotation == list[DataContainer]
            ):
                args[keys[0]] = node_inputs
            else:
                args = {**args, **dict_inputs}

            # once all the nodes are migrated to the new node api, remove the if condition
            if len(sig.parameters) == 2 and sig.parameters[keys[1]].annotation == dict:
                args[keys[1]] = func_params
            else:
                for param, value in func_params.items():
                    if param in sig.parameters:
                        args[param] = value

            print("calling node with args keys:", args.keys(), flush=True)

            ##########################
            # calling the node function
            ##########################
            dc_obj = func(**args)  # DataContainer object from node
            ##########################
            # end calling the node function
            ##########################

            if isinstance(
                dc_obj, DataContainer
            ):  # some special nodes like LOOP return dict instead of `DataContainer`
                dc_obj.validate()  # Validate returned DataContainer object
            result = get_frontend_res_obj_from_result(
                dc_obj
            )  # Response object to send to FE

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

            JobService().post_job_result(job_id, result) # post result to the job service
            return result
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
            print("error occured while running the node", flush=True)
            print(traceback.format_exc(), flush=True)
            raise e

    return wrapper


def reactflow_to_networkx(elems: list[Any], edges: list[Any]):
    nx_graph: nx.DiGraph = nx.DiGraph()
    for i in range(len(elems)):
        el = elems[i]
        node_id = el["id"]
        data = el["data"]
        cmd = el["data"]["func"]
        ctrls: dict[str, Any] = data["ctrls"] if "ctrls" in data else {}
        inputs: dict[str, Any] = data["inputs"] if "inputs" in data else {}
        label: dict[str, Any] = data["label"] if "label" in data else {}
        nx_graph.add_node(  # type:ignore
            node_id,
            pos=(el["position"]["x"], el["position"]["y"]),
            id=el["id"],
            ctrls=ctrls,
            inputs=inputs,
            label=label,
            cmd=cmd,
        )

    for i in range(len(edges)):
        e = edges[i]
        _id = e["id"]
        u = e["source"]
        v = e["target"]
        label = e["sourceHandle"]
        nx_graph.add_edge(u, v, label=label, id=_id)  # type:ignore

    nx_draw(nx_graph, with_labels=True)

    return nx_graph
