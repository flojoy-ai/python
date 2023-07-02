from typing import Callable, Any, Optional
from .data_container import *
from .flojoy_python import *
from .job_result_builder import *
from .flojoy_instruction import *
from .plotly_utils import *
from .module_scraper import *
from .job_result_utils import *
from .data_container import *
from .utils import *

def flojoy(
    original_function: Callable[..., DataContainer | dict[str, Any]] | None = None,
    *,
    node_type: Optional[str] = None,
    deps: Optional[dict[str, str]] = None,
    inject_node_metadata: bool = False
) -> Callable[..., DataContainer | dict[str, Any]]: ...