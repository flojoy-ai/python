import re
import os
import json
import yaml
import traceback
import numpy as np
import networkx as nx
from rq.job import Job  # type:ignore
from pathlib import Path
from functools import wraps
from .data_container import DataContainer
from .utils import PlotlyJSONEncoder, dump_str
from networkx.drawing.nx_pylab import draw as nx_draw  # type: ignore
from typing import Union, cast, Any, Literal, Callable, List, Optional, Type
from .job_result_utils import get_frontend_res_obj_from_result, get_dc_from_result
from .utils import redis_instance, send_to_socket


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


def js_to_json(s: str):
    """
    Converts an ES6 JS file with a single JS Object definition to JSON
    """
    split = s.split("=")[1]
    clean = split.replace("\n", "").replace("'", "").replace(",}", "}").rstrip(";")
    single_space = "".join(clean.split())
    dbl_quotes = re.sub(r"(\w+)", r'"\1"', single_space).replace('""', '"')
    rm_comma = dbl_quotes.replace("},}", "}}")

    return json.loads(rm_comma)


def get_parameter_manifest() -> dict[str, Any]:
    root = get_flojoy_root_dir()
    f = open(os.path.join(root, "src/data/manifests-latest.json"))
    param_manifest = json.load(f)
    return param_manifest["parameters"]


def fetch_inputs(
    previous_job_ids: list[str], mock: bool = False
) -> list[DataContainer]:
    """
    Queries Redis for job results

    Parameters
    ----------
    previous_job_ids : list of Redis job IDs that directly precede this node

    Returns
    -------
    inputs : list of DataContainer objects
    """
    if mock is True:
        return [DataContainer(x=np.linspace(0, 10, 100))]

    inputs: list[DataContainer] = []

    try:
        for prev_job_id in previous_job_ids:
            print("fetching input from prev job id:", prev_job_id)
            job = Job.fetch(prev_job_id, connection=redis_instance)  # type:ignore
            job_result: dict[str, Any] = job.result  # type:ignore
            result = get_dc_from_result(job_result)
            print(
                "fetch input from prev job id:",
                prev_job_id,
                " result:",
                dump_str(result, limit=100),
            )
            if result is not None:
                inputs.append(result)
    except Exception:
        print(traceback.format_exc())

    return inputs


def get_redis_obj(id: str) -> dict[str, Any]:
    get_obj = redis_instance.get(id)
    parse_obj: dict[str, Any] = json.loads(get_obj) if get_obj is not None else {}
    return parse_obj


def parse_array(str_value: str) -> List[Union[int, float, str]]:
    if not str_value:
        return []

    val_list = [val.strip() for val in str_value.split(",")]
    # First try to cast into int, then float, then keep as string if all else fails
    for t in [int, float, str]:
        try:
            val = list(map(t, val_list))
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
            mock = False
            previous_job_ids = cast(list[str], kwargs.get("previous_job_ids", []))
            ctrls = cast(
                Union[dict[str, dict[str, str]], None], kwargs.get("ctrls", None)
            )
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
            # Get default command paramaters
            default_params: dict[str, Any] = {}
            func_params = {}
            pm = get_parameter_manifest()
            if FN in pm:
                for param in pm[FN]:
                    default_params[param] = pm[FN][param]["default"]
                # Get command parameters set by the user through the control panel
                func_params = {}
                if ctrls is not None:
                    for key, input in ctrls.items():
                        param = input["param"]
                        val = input["value"]
                        func_params[param] = format_param_value(
                            val, pm[FN][param]["type"]
                        )
                # Make sure that function parameters set is fully loaded
                # If function is missing a parameter, fill-in with default value
                for key in default_params.keys():
                    if key not in func_params.keys():
                        func_params[key] = default_params[key]

            func_params["jobset_id"] = jobset_id
            func_params["type"] = "default"
            func_params["node_id"] = node_id
            func_params["job_id"] = job_id

            print("executing node_id:", node_id, "previous_job_ids:", previous_job_ids)
            print(node_id, " params: ", json.dumps(func_params, indent=2))
            node_inputs = fetch_inputs(previous_job_ids, mock)

            # running the node
            dc_obj = func(node_inputs, func_params)  # DataContainer object from node
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
            print("final result:", dump_str(result, limit=100))
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
            print("error occured while running the node")
            print(traceback.format_exc())
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
