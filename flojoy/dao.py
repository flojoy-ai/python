import json
from redis import Redis
import os
import numpy as np
import pandas as pd
from typing import Any, cast
from threading import Lock

MAX_LIST_SIZE = 1000


def get_memory():
    return Dao.get_instance()

"""
This class is a Singleton that acts as a in-memory datastorage

IMPORTANT: The commented code should not be removed, as it acts as a reference for the future
in case we need to implement a Redis based datastorage
"""
class Dao:

    _instance = None
    _init_lock = Lock()
    _dict_lock = Lock()

    @classmethod
    def get_instance(cls):
        with cls._init_lock:
            if Dao._instance is None:
                Dao._instance = Dao()
            return Dao._instance

    def __init__(self):
        self.storage = {}

    def check_if_valid(self, result, expected_type):
        if not isinstance(result, expected_type):
            raise ValueError(
                f"Expected {expected_type} type, but got {type(result)} instead!"
            )

    def set_np_array(self, memo_key: str, value: np.ndarray):
        # encoded = self.serialize_np(value)
        # self.storage[memo_key] = encoded
        with self._dict_lock:
            self.storage[memo_key] = value

    def set_pandas_dataframe(self, key: str, dframe: pd.DataFrame):
        # encode = dframe.to_json()
        #  self.storage[key] = encode
        with self._dict_lock:
            self.storage[key] = dframe

    def set_str(self, key: str, value: str):
        with self._dict_lock:
            self.storage[key] = value

    def get_pd_dataframe(self, key: str) -> pd.DataFrame | None:
        encoded = self.storage.get(key, None)
        if encoded is None:
            return None
        self.check_if_valid(encoded, pd.DataFrame)
        # decode = encoded.decode("utf-8") if encoded is not None else ""
        # read_json = pd.read_json(decode)
        return encoded.head()

    def get_np_array(self, memo_key: str, np_meta_data: dict[str, str]) -> np.ndarray | None:
        encoded = self.storage.get(memo_key, None)
        if encoded is None:
            return None
        self.check_if_valid(encoded, np.ndarray)
        return encoded

    def get_str(self, key: str) -> str | None:
        encoded = self.storage.get(key, None)
        if encoded is None:
            return None
        # return encoded.decode("utf-8") if encoded is not None else None
        self.check_if_valid(encoded, str)
        return encoded
        
    def get_obj(self, key: str) -> dict[str, Any] | None:
        r_obj = self.storage.get(key, None)
        if r_obj is None:
            return None
        # if r_obj:
        #     return cast(dict[str, Any], json.loads(r_obj))
        self.check_if_valid(r_obj, dict)
        return r_obj

    def set_obj(self, key: str, value: dict[str, Any]):
        # dump = json.dumps(value)
        with self._dict_lock:
            self.storage[key] = value

    def delete_object(self, key: str):
        with self._dict_lock:
            self.storage.pop(key)

    def remove_item_from_set(self, key: str, item: Any):
        res = self.storage.get(key, None)
        self.check_if_valid(res, set)
        with self._dict_lock:
            res.remove(item)

    def add_to_set(self, key: str, value: Any):
        res = self.storage.get(key, None)
        if res is None:
            res = set()
            res.add(value)
            self.storage[key] = res
            return 
        self.check_if_valid(res, set)
        with self._dict_lock:
            res.add(value)

    def get_set_list(self, key: str) -> list[Any] | None:
        res = self.storage.get(key, None)
        if res is None:
            return None
        self.check_if_valid(res, set)
        return list(res)

    def serialize_np(self, np_array: np.ndarray):
        return np_array.ravel().tostring()

    def desirialize_np(self, encoded: bytes, np_meta_data: dict[str, str]):
        d_type = np_meta_data.get("d_type", "")
        dimensions = np_meta_data.get("dimensions", [])
        shapes_in_int = [int(shape) for shape in dimensions]
        return np.fromstring(encoded, dtype=d_type).reshape(*shapes_in_int)
