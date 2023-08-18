import os
from dotenv import dotenv_values  # type:ignore
from .config import FlojoyConfig, logger
import requests
from typing import Any
from pydantic import BaseModel
import json

env_vars = dotenv_values("../.env")
port = env_vars.get("VITE_BACKEND_PORT", "5392")
BACKEND_URL = os.environ.get("BACKEND_URL", f"http://127.0.0.1:{port}")




class NodeResults(BaseModel):
    cmd: str
    id: str
    result: dict[str, Any]


class ModalConfig(dict):
    showModal: bool | None
    title: str | None
    messages: str | None
    id : str | None

class SocketData(dict):
    SYSTEM_STATUS: str | None = None
    NODE_RESULTS: NodeResults | None = None
    RUNNING_NODE: str | None = None
    FAILED_NODES: dict[str, str] | None = None
    PRE_JOB_OP: dict[str, Any] | None = None
    jobsetId: str = ""
    MODAL_CONFIG : ModalConfig

    def __init__(
        self,
        jobset_id: str,
        sys_status: str | None = None,
        failed_nodes: dict[str, str] | None = None,
        running_node: str = "",
        dict_item: dict[str, Any] = {},
        modal_config: ModalConfig | None = None
    ):
        self["jobsetId"] = jobset_id
        if sys_status:
            self["SYSTEM_STATUS"] = sys_status
        self["type"] = "worker_response"
        self["FAILED_NODES"] = failed_nodes or {}
        self["RUNNING_NODE"] = running_node
        self["MODAL_CONFIG"] = modal_config or ModalConfig(showModal=False)
        for k, item in dict_item.items():
            self[k] = item

    def __setitem__(self, __key: Any, __value: Any) -> None:
        super().__setattr__(__key, __value)
        return super().__setitem__(__key, __value)
      
    def _to_json(self):
        dumps = json.dumps(self.copy())
        return dumps

def send_to_socket(data: SocketData):
    if FlojoyConfig.get_instance().is_offline:
        return
    try:
      
      print(" send to socket: ", "backend_url: ", BACKEND_URL, data, " json: ", data._to_json(), flush=True)
      # logger.debug("posting data to socket:", f"{BACKEND_URL}/worker_response")
      res = requests.post(f"{BACKEND_URL}/worker_response", json=data._to_json())
      print(" res form send to socket: ", res, flush=True)
    except Exception as e:
      print("error in send to socket ", e,  flush=True)
