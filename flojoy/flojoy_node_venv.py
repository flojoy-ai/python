"""
This module provides a decorator that allows a function to be executed in a virtual environment.
The decorator will create a virtual environment if it does not exist, and install the pip dependencies
specified in the decorator arguments. The decorator will also install the pip dependencies if the
virtual environment exists but does not contain the pip dependencies.

Example usage:

```python
from flojoy import flojoy, run_in_venv

@flojoy
@run_in_venv(pip_dependencies=["torch==2.0.1", "torchvision==0.15.2"])
def TORCH_NODE(default: Matrix) -> Matrix:
    import torch
    import torchvision
    # Do stuff with torch
    ...
    return Matrix(...)

"""
import hashlib
import importlib.metadata
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import traceback
import venv
from functools import wraps
import cloudpickle

from .utils import FLOJOY_CACHE_DIR

__all__ = ["run_in_venv"]


class MultiprocessingExecutableContextManager:
    """Temporarily change the executable used by multiprocessing."""
    def __init__(self, executable_path):
        self.original_executable_path = sys.executable
        self.executable_path = executable_path
        # We need to save the original start method 
        # because it is set to "fork" by default on Linux while we ALWAYS want spawn
        self.original_start_method = multiprocessing.get_start_method()

    def __enter__(self):
        if(self.original_start_method != "spawn"):
            multiprocessing.set_start_method("spawn", force=True)
        multiprocessing.set_executable(self.executable_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if(self.original_start_method != "spawn"):
            multiprocessing.set_start_method(self.original_start_method, force=True)
        multiprocessing.set_executable(self.original_executable_path)

class SwapSysPath:
    """Temporarily swap the sys.path of the child process with the sys.path of the parent process."""
    def __init__(self, venv_executable):
        self.new_path = _get_venv_syspath(venv_executable)
        self.old_path = None

    def __enter__(self):
        self.old_path = sys.path
        sys.path = self.new_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.path = self.old_path


def _install_pip_dependencies(
        venv_executable: os.PathLike,
        pip_dependencies: tuple[str],
        verbose: bool = False
    ):
    """Install pip dependencies into the virtual environment."""
    command = [venv_executable, "-m", "pip", "install"]
    if(not verbose):
        command += ["-q", "-q"]  
    command += list(pip_dependencies)
    subprocess.check_call(command)

def _get_venv_syspath(venv_executable: os.PathLike) -> list[str]:
    """Get the sys.path of the virtual environment."""
    command = [venv_executable, "-c", "import sys\nprint(sys.path)"]
    cmd_output = subprocess.run(command, check=True, capture_output=True, text=True)
    return eval(cmd_output.stdout)

class PickleableFunctionWithPipeIO:
    """A wrapper for a function that can be pickled and executed in a child process."""
    def __init__(self, func, child_conn, venv_executable):
        self._func_serialized = cloudpickle.dumps(func)
        self._child_conn = child_conn
        self._venv_executable = venv_executable

    def __call__(self, *args, **kwargs):
        fn = cloudpickle.loads(self._func_serialized)
        args = [cloudpickle.loads(arg) for arg in args]
        kwargs = {key: cloudpickle.loads(value) for key, value in kwargs.items()}
        with SwapSysPath(venv_executable=self._venv_executable):
            try:
                result = fn(*args, **kwargs)
            except Exception as e:
                result = (e, traceback.format_exception(type(e), e, e.__traceback__))
        self._child_conn.send(result)

def _get_venv_executable_path(venv_path: os.PathLike | str) -> os.PathLike | str:
    """Get the path to the python executable of the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")

def _get_venv_cache_dir():
    return os.path.join(FLOJOY_CACHE_DIR, "flojoy_node_venv")

def run_in_venv(pip_dependencies: list[str] = [], verbose: bool = False):
    """A decorator that allows a function to be executed in a virtual environment.
    
    Args:
        pip_dependencies (list[str]): A list of pip dependencies to install into the virtual environment. Defaults to [].
        verbose (bool): Whether to print the pip install output. Defaults to False.

    Example usage:
    ```python
    from flojoy import flojoy, run_in_venv
    
    @flojoy
    @run_in_venv(pip_dependencies=["torch==2.0.1", "torchvision==0.15.2"])
    def TORCH_NODE(default: Matrix) -> Matrix:
        import torch
        import torchvision
        # Do stuff with torch
        ...
        return Matrix(...)
    """
    # Pre-pend flojoy and cloudpickle as mandatory pip dependencies
    packages_dict = {package.name: package.version for package in importlib.metadata.distributions()}
    pip_dependencies = sorted([
        f"flojoy=={packages_dict['flojoy']}",
        f"cloudpickle=={packages_dict['cloudpickle']}"] \
        + pip_dependencies
    )
    # Get the root directory for the virtual environments
    venv_cache_dir = _get_venv_cache_dir()
    os.makedirs(venv_cache_dir, exist_ok=True)
    # Generate a path-safe hash of the pip dependencies
    # this prevents the duplication of virtual environments
    pip_dependencies_hash = hashlib.sha256("".join(pip_dependencies).encode()).hexdigest()
    venv_path = os.path.join(venv_cache_dir, f"{pip_dependencies_hash}")
    venv_executable = _get_venv_executable_path(venv_path)
    # Create the node_env virtual environment if it does not exist
    if not os.path.exists(venv_path):
        venv.create(venv_path, with_pip=True)
        # Install the pip dependencies into the virtual environment
        if pip_dependencies:
            try:
                _install_pip_dependencies(
                    venv_executable=venv_executable,
                    pip_dependencies=tuple(pip_dependencies),
                    verbose=verbose
                )
            except subprocess.CalledProcessError as e:
                shutil.rmtree(venv_path)
                logging.error(f"Failed to install pip dependencies into virtual environment from the provided list: {pip_dependencies}")
                raise e

    # Define the decorator
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Serialize function arguments using cloudpickle
            parent_conn, child_conn = multiprocessing.Pipe()
            args = [cloudpickle.dumps(arg) for arg in args]
            kwargs = {key: cloudpickle.dumps(value) for key, value in kwargs.items()}
            pickleable_func_with_pipe = PickleableFunctionWithPipeIO(func, child_conn, venv_executable)
            # Start the context manager that will change the executable used by multiprocessing
            with MultiprocessingExecutableContextManager(venv_executable):
                # Create a new process that will run the Python code
                process = multiprocessing.Process(target=pickleable_func_with_pipe, args=args, kwargs=kwargs)
                # Start the process
                process.start()
                # Fetch result from the child process
                serialized_result = parent_conn.recv_bytes()
                # Wait for the process to finish
                process.join()
            # Check if the process sent an exception with a traceback
            result = cloudpickle.loads(serialized_result)
            if isinstance(result, tuple) and isinstance(result[0], Exception):
                # Fetch exception and formatted traceback (list[str])
                exception, tcb = result
                # Reraise an exception with the same class
                logging.error(f"Error in child process \n{''.join(tcb)}")
                raise exception
            return result
        return wrapper
    return decorator
