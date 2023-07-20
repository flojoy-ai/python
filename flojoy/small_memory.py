from .dao import Dao
from typing import Any

__all__ = ["SmallMemory"]


class SmallMemory:
    """
    SmallMemory - available during jobset execution - intended to be used ONLY inside node functions
    """

    """_______________________________________________________________________

    Methods used inside of node function:
    """
    tracing_key = "ALL_MEMORY_KEYS"

    def write_to_memory(self, job_id: str, key: str, value: Any):
        memory_key = f"{job_id}-{key}"
        value_type_key = f"{memory_key}_value_type_key"
        meta_data = {}
        s = str(type(value))
        v_type = s.split("'")[1]
        match v_type:
            case "numpy.ndarray":
                array_dtype = str(value.dtype)
                meta_data["type"] = "np_array"
                meta_data["d_type"] = array_dtype
                meta_data["dimensions"] = value.shape
                Dao.get_instance().set_obj(value_type_key, meta_data)
                Dao.get_instance().set_np_array(memory_key, value)
            case "pandas.core.frame.DataFrame":
                meta_data["type"] = "pd_dframe"
                Dao.get_instance().set_obj(value_type_key, meta_data)
                Dao.get_instance().set_pandas_dataframe(memory_key, value)
            case "str" | "numpy.float64":
                meta_data["type"] = "string"
                Dao.get_instance().set_obj(value_type_key, meta_data)
                Dao.get_instance().set_str(memory_key, value)
            case "dict":
                meta_data["type"] = "dict"
                Dao.get_instance().set_obj(value_type_key, meta_data)
                Dao.get_instance().set_obj(memory_key, value)
            case _:
                raise ValueError(
                    f"SmallMemory currently does not support '{v_type}' type data!"
                )

    def read_memory(self, job_id: str, key: str):
        """
        Reads object stored in internal DB by the given key. The memory is job specific.
        """
        memory_key = f"{job_id}-{key}"
        value_type_key = f"{memory_key}_value_type_key"
        meta_data = Dao.get_instance().get_obj(value_type_key)
        meta_type = meta_data.get("type")
        match meta_type:
            case "string":
                return Dao.get_instance().get_str(memory_key)
            case "dict":
                return Dao.get_instance().get_obj(memory_key)
            case "np_array":
                return Dao.get_instance().get_np_array(memory_key, meta_data)
            case "pd_dframe":
                return Dao.get_instance().get_pd_dataframe(memory_key)
            case _:
                return None

    def delete_object(self, job_id: str, key: str):
        """
        Removes object stored in internal DB by the given key. The memory is job specific.
        """
        memory_key = f"{job_id}-{key}"
        return Dao.get_instance().delete_object(memory_key)

    """_______________________________________________________________________
    """
