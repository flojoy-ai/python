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
from typing import Callable

import hashlib
from contextlib import contextmanager
import importlib.metadata
import inspect
import logging
import multiprocessing
import multiprocessing.connection
import os
import shutil
import subprocess
import sys
import traceback
import venv
from functools import wraps
import cloudpickle

from .utils import FLOJOY_CACHE_DIR
from .socket_utils import SocketData, ModalConfig, send_to_socket
from .logging import LogPipe
import json

__all__ = ["run_in_venv"]


@contextmanager
def swap_sys_path(venv_executable: os.PathLike, extra_sys_path: list[str] | None = None):
    """Temporarily swap the sys.path of the child process with the sys.path of the parent process."""
    old_path = sys.path
    try:
        new_path = _get_venv_syspath(venv_executable)
        extra_sys_path = [] if extra_sys_path is None else extra_sys_path
        sys.path = new_path + extra_sys_path
        yield
    finally:
        sys.path = old_path

def stream_response(proc):
    """Stream the output of a subprocess."""
    while True:
        line = proc.stdout.readline() or proc.stderr.readline()
        if not line:
            break
        yield line

def _install_pip_dependencies(
    venv_executable: os.PathLike, pip_dependencies: tuple[str], socket_data: SocketData, verbose: bool = False,
    
):
    """Install pip dependencies into the virtual environment."""
    command = [venv_executable, "-m", "pip", "install"]
    if not verbose:
        command += ["-q", "-q"]
    command += list(pip_dependencies)
    with LogPipe(logging.INFO) as pipe_stdout, LogPipe(logging.ERROR) as pipe_stderr:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while proc.poll() is None:
            stream = stream_response(proc)
            for line in stream:
                socket_data.MODAL_CONFIG["messages"] = line.decode(encoding="utf-8")
                send_to_socket(socket_data)
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=pipe_stdout.buffer.getvalue(),
            stderr=pipe_stderr.buffer.getvalue(),
        )


def _get_venv_syspath(venv_executable: os.PathLike) -> list[str]:
    """Get the sys.path of the virtual environment."""
    command = [venv_executable, "-c", "import sys\nprint(sys.path)"]
    cmd_output = subprocess.run(command, check=True, capture_output=True, text=True)
    return eval(cmd_output.stdout)


class PickleableFunctionWithPipeIO:
    """A wrapper for a function that can be pickled and executed in a child process."""

    def __init__(
        self,
        func: Callable,
        child_conn: multiprocessing.connection.Connection,
        venv_executable: str,
    ):
        self._func_serialized = cloudpickle.dumps(func)
        func_module_path = os.path.dirname(os.path.realpath(inspect.getabsfile(func)))
        # Check that the function is in a directory indeed
        self._extra_sys_path = (
            [func_module_path] if os.path.isdir(func_module_path) else None
        )
        self._child_conn = child_conn
        self._venv_executable = venv_executable

    def __call__(self, *args_serialized, **kwargs_serialized):
        with swap_sys_path(
            venv_executable=self._venv_executable, extra_sys_path=self._extra_sys_path
        ):
            try:
                fn = cloudpickle.loads(self._func_serialized)
                args = [cloudpickle.loads(arg) for arg in args_serialized]
                kwargs = {
                    key: cloudpickle.loads(value)
                    for key, value in kwargs_serialized.items()
                }
                serialized_result = cloudpickle.dumps(fn(*args, **kwargs))
            except Exception as e:
                # Not all exceptions are expected to be picklable
                # so we clone their traceback and send our own custom type of exception
                exc = ChildProcessError(
                    f"Child process failed with an exception of type {type(e)}."
                ).with_traceback(e.__traceback__)
                serialized_result = cloudpickle.dumps(
                    (exc, traceback.format_exception(type(e), e, e.__traceback__))
                )
        self._child_conn.send_bytes(serialized_result)


def _get_venv_executable_path(venv_path: os.PathLike | str) -> os.PathLike | str:
    """Get the path to the python executable of the virtual environment."""
    if sys.platform == "win32":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def _get_venv_cache_dir():
    return os.path.join(FLOJOY_CACHE_DIR, "flojoy_node_venv")


def run_in_venv(pip_dependencies: list[str] | None = None, verbose: bool = False):
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
    jobset_id = os.environ.get("FC_JOBSET_ID", "")
    socket_data = SocketData(jobset_id=jobset_id, modal_config=ModalConfig(showModal=True))
    if pip_dependencies is None:
        pip_dependencies = []
    # Pre-pend flojoy and cloudpickle as mandatory pip dependencies
    packages_dict = {
        package.name: package.version for package in importlib.metadata.distributions()
    }
    pip_dependencies = sorted(
        [
            f"flojoy=={packages_dict['flojoy']}",
            f"cloudpickle=={packages_dict['cloudpickle']}",
        ]
        + pip_dependencies
    )
    # Get the root directory for the virtual environments
    venv_cache_dir = _get_venv_cache_dir()
    os.makedirs(venv_cache_dir, exist_ok=True)
    # Generate a path-safe hash of the pip dependencies
    # this prevents the duplication of virtual environments
    pip_dependencies_hash = hashlib.md5(
        "".join(sorted(pip_dependencies)).encode()
    ).hexdigest()[:8]
    venv_path = os.path.join(venv_cache_dir, f"{pip_dependencies_hash}")
    venv_executable = _get_venv_executable_path(venv_path)
    socket_data.MODAL_CONFIG["title"] = "Node virtual environment"
    # Create the node_env virtual environment if it does not exist
    if not os.path.exists(venv_path):
        socket_data.MODAL_CONFIG['messages'] = "Some node requires running in a separate virtual environment.."
        print(" posting data line 204: ", socket_data, type(socket_data._to_json()), flush=True)
        send_to_socket(socket_data)

        socket_data.MODAL_CONFIG["messages"] = f"Creating virtual environment at {venv_path}"
        print(" posting data line 207: ", socket_data, type(socket_data._to_json()), flush=True)
        send_to_socket(socket_data)
        venv.create(venv_path, with_pip=True)
        # Install the pip dependencies into the virtual environment
        if pip_dependencies:
            try:
                socket_data.MODAL_CONFIG["messages"] = f"Installing {', '.join(pip_dependencies)} packages into virtual environment..."
                print(" posting data line 214: ", socket_data, type(socket_data._to_json()), flush=True)
                send_to_socket(socket_data)
                _install_pip_dependencies(
                    venv_executable=venv_executable,
                    pip_dependencies=tuple(pip_dependencies),
                    socket_data=socket_data,
                    verbose=verbose,
                    
                )
            except subprocess.CalledProcessError as e:
                shutil.rmtree(venv_path)
                logging.error(
                    f"[ _install_pip_dependencies ] Failed to install pip dependencies into virtual environment from the provided list: {pip_dependencies}. The virtual environment under {venv_path} has been deleted."
                )
                # Log every line of e.stderr
                for line in e.stderr.decode().splitlines():
                    socket_data.MODAL_CONFIG["messages"] = "[error]: " + line
                    send_to_socket(socket_data)
                    logging.error(f"[ _install_pip_dependencies ] {line}")
                socket_data.MODAL_CONFIG["messages"] = "Failed to install pip dependencies into virtual environment"
                socket_data["SYSTEM_STATUS"] = "❌ Failed to install pip dependencies into virtual environment"
                send_to_socket(socket_data)
                raise e

    
    
    # Define the decorator
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a new multiprocessing context for the parent process in "spawn" mode
            parent_mp_context = multiprocessing.get_context("spawn")
            parent_conn, child_conn = parent_mp_context.Pipe()
            # Serialize function arguments using cloudpickle
            args_serialized = [cloudpickle.dumps(arg) for arg in args]
            kwargs_serialized = {
                key: cloudpickle.dumps(value) for key, value in kwargs.items()
            }
            pickleable_func_with_pipe = PickleableFunctionWithPipeIO(
                func, child_conn, venv_executable
            )
            # Create a new multiprocessing context for the child process in "spawn" mode
            # while setting its executable to the virtual environment python executable
            child_mp_context = multiprocessing.get_context("spawn")
            child_mp_context.set_executable(venv_executable)
            # Create a new process that will run the Python code
            process = child_mp_context.Process(
                target=pickleable_func_with_pipe,
                args=args_serialized,
                kwargs=kwargs_serialized,
            )
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
                logging.error(
                    f"[ run_in_venv ] Error in child process with the following traceback:\n{''.join(tcb)}"
                )
                raise exception
            return result

        return wrapper

    return decorator
