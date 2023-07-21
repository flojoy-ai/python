import numpy as np
import pandas as pd
from typing import Any
from threading import Lock
from .data_container import DCNpArrayType

MAX_LIST_SIZE = 1000

"""
Used by clients to create a new instance of the datastorage
"""


def create_storage():
    return Dao.get_instance()


"""
This class is a Singleton that acts as a in-memory datastorage

IMPORTANT: The commented code should not be removed, as it acts as a reference for the future
in case we need to implement a Redis based datastorage
"""


class Dao:
    _instance = None
    _init_lock = Lock()
    _dict_sm_lock = Lock()  # dict small memory lock
    _dict_job_lock = Lock()  # dict job lock

    @classmethod
    def get_instance(cls):
        with cls._init_lock:
            if Dao._instance is None:
                Dao._instance = Dao()
            return Dao._instance

    def __init__(self):
        self.storage: dict[str, Any] = {}  # small memory
        self.job_results: dict[str, Any] = {}

    """
    METHODS FOR JOB RESULTS
    """

    def get_job_result(self, job_id: str) -> Any | None:
        res = self.job_results.get(job_id, None)
        if res is None:
            raise ValueError(f"Job result with id {job_id} does not exist")
        return res

    def post_job_result(self, job_id: str, result: Any):
        with self._dict_job_lock:
            self.job_results[job_id] = result

    def clear_job_results(self):
        with self._dict_job_lock:
            self.job_results.clear()

    def job_exists(self, job_id: str) -> bool:
        return job_id in self.job_results.keys()

    def delete_job(self, job_id: str):
        with self._dict_job_lock:
            self.job_results.pop(job_id, None)

    """
    METHODS FOR SMALL MEMORY
    """

    def clear_small_memory(self):
        with self._dict_sm_lock:
            self.storage.clear()

    def check_if_valid(self, result: Any | None, expected_type: Any):
        if result is not None and not isinstance(result, expected_type):
            raise ValueError(
                f"Expected {expected_type} type, but got {type(result)} instead!"
            )

    def set_np_array(self, memo_key: str, value: DCNpArrayType):
        with self._dict_sm_lock:
            self.storage[memo_key] = value

    def set_pandas_dataframe(self, key: str, dframe: pd.DataFrame):
        with self._dict_sm_lock:
            self.storage[key] = dframe

    def set_str(self, key: str, value: str):
        with self._dict_sm_lock:
            self.storage[key] = value

    def get_pd_dataframe(self, key: str) -> pd.DataFrame | None:
        encoded = self.storage.get(key, None)
        self.check_if_valid(encoded, pd.DataFrame)
        return encoded

    def get_np_array(self, memo_key: str) -> DCNpArrayType | None:
        encoded = self.storage.get(memo_key, None)
        self.check_if_valid(encoded, np.ndarray)
        return encoded

    def get_str(self, key: str) -> str | None:
        encoded = self.storage.get(key, None)
        return encoded

    def get_obj(self, key: str) -> dict[str, Any] | None:
        r_obj = self.storage.get(key, None)
        self.check_if_valid(r_obj, dict)
        return r_obj

    def set_obj(self, key: str, value: dict[str, Any]):
        with self._dict_sm_lock:
            self.storage[key] = value

    def delete_object(self, key: str):
        with self._dict_sm_lock:
            self.storage.pop(key)

    def remove_item_from_set(self, key: str, item: Any):
        res = self.storage.get(key, None)
        self.check_if_valid(res, set)
        if not res:
            return
        with self._dict_sm_lock:
            res.remove(item)

    def add_to_set(self, key: str, value: Any):
        res: set[Any] | None = self.storage.get(key, None)
        if res is None:
            res = set()
            res.add(value)
            self.storage[key] = res
            return
        self.check_if_valid(res, set)
        with self._dict_sm_lock:
            res.add(value)

    def get_set_list(self, key: str) -> list[Any] | None:
        res = self.storage.get(key, None)
        if res is None:
            return None
        self.check_if_valid(res, set)
        return list(res)
