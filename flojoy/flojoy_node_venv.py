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
from time import sleep
from typing import Callable, Optional, TextIO

import hashlib
from contextlib import ExitStack, contextmanager
import importlib.metadata
import inspect
import logging
import multiprocessing
import multiprocessing.connection
import os
import shutil
import re
import subprocess
import sys
import traceback
import venv
from functools import wraps
import cloudpickle

from .utils import FLOJOY_CACHE_DIR
from .logging import LogPipe, LogPipeMode

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


def _install_pip_dependencies(
    venv_executable: os.PathLike,
    pip_dependencies: tuple[str],
    logger: logging.Logger,
    verbose: bool = False
):
    """Install pip dependencies into the virtual environment."""
    command = [venv_executable, "-m", "pip", "install"]
    if not verbose:
        command += ["-q", "-q"]
    command += list(pip_dependencies)
    with ExitStack() as stack:
        logpipe_stderr = stack.enter_context(LogPipe(logger, log_level=logging.ERROR, mode=LogPipeMode.SUBPROCESS))
        logpipe_stdout = stack.enter_context(LogPipe(logger, log_level=logging.DEBUG, mode=LogPipeMode.SUBPROCESS))
        proc = subprocess.Popen(command, stdout=logpipe_stdout, stderr=logpipe_stderr)
        proc.wait()
        sleep(5)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            command,
            output=logpipe_stdout.buffer.getvalue(),
        )


def _get_venv_syspath(venv_executable: os.PathLike) -> list[str]:
    """Get the sys.path of the virtual environment."""
    command = [venv_executable, "-c", "import sys\nprint(sys.path)"]
    cmd_output = subprocess.run(command, check=True, capture_output=True, text=True)
    return eval(cmd_output.stdout)

@contextmanager
def redirect_streams(log_pipe_writer: TextIO):
    import sys
    try:
        sys.stdout = log_pipe_writer
        yield
    finally:
        sys.stdout = sys.__stdout__

class PickleableFunctionWithPipeIO:
    """A wrapper for a function that can be pickled and executed in a child process."""

    def __init__(
        self,
        func: Callable,
        child_conn: multiprocessing.connection.Connection,
        log_pipe_writer: TextIO,
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
        self.log_pipe_writer = log_pipe_writer

    def __call__(self, *args_serialized, **kwargs_serialized):
        with redirect_streams(log_pipe_writer=self.log_pipe_writer):
            with swap_sys_path(venv_executable=self._venv_executable, extra_sys_path=self._extra_sys_path):
                try:
                    fn = cloudpickle.loads(self._func_serialized)
                    args = [cloudpickle.loads(arg) for arg in args_serialized]
                    kwargs = {
                        key: cloudpickle.loads(value)
                        for key, value in kwargs_serialized.items()
                    }
                    # Capture logs here too
                    output = fn(*args, **kwargs)
                    serialized_result = cloudpickle.dumps(output)
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

def _get_decorated_function_name(decorator_name: str) -> Optional[str]:
    stack = inspect.stack()
    # Fetch the frame for which the @no_op_decorator is present
    lineno, file_path = None, None
    for frame in stack:
        if(any([f"@{decorator_name}" in context for context in frame.code_context])):
            lineno = frame.lineno
            file_path = os.path.realpath(frame.filename)
            break
    if lineno is None or file_path is None:
        return None 
    if not os.path.exists(file_path):
        return None
    # Read the file
    with open(file_path) as f:
        lines = f.readlines()
    func_name = None
    # Fetch the function name after lineno, i.e. the first function name after @no_op_decorator
    to_search = lines[lineno:]
    for line in to_search:
        if "def" in line:
            func_name = re.findall(r"def\s+([^\s(]+)", line)[0]
            break
    return func_name



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
    # Create the node_env virtual environment if it does not exist
    if not os.path.exists(venv_path):
        venv.create(venv_path, with_pip=True)
    # Get the name of the module in the parent frame
    # this is needed to get the path of the module
    # that the function is defined in
    func_name = _get_decorated_function_name(decorator_name="run_in_venv")
    logger = logging.getLogger(func_name)
    # Install the pip dependencies into the virtual environment
    if pip_dependencies:
        try:
            _install_pip_dependencies(
                venv_executable=venv_executable,
                pip_dependencies=tuple(pip_dependencies),
                logger=logger,
                verbose=verbose,
            )
        except subprocess.CalledProcessError as e:
            shutil.rmtree(venv_path)
            logger.error(
                f"Failed to install pip dependencies into virtual environment from the provided list: {pip_dependencies}. The virtual environment under {venv_path} has been deleted."
            )
            # Log every line of e.stderr
            for line in e.stderr.decode().splitlines():
                logging.error(f"{line}")
            raise e

    # Define the decorator
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate a new multiprocessing context for the parent process in "spawn" mode
            parent_mp_context = multiprocessing.get_context("spawn")
            parent_conn, child_conn = parent_mp_context.Pipe()
            # Create a new multiprocessing context for the child process in "spawn" mode
            # while setting its executable to the virtual environment python executable
            child_mp_context = multiprocessing.get_context("spawn")
            child_mp_context.set_executable(venv_executable)
            func_name = func.__name__
            with LogPipe(logging.getLogger(func_name), logging.INFO, LogPipeMode.MP_SPAWN) as log_pipe:
                # Serialize function arguments using cloudpickle
                pickleable_func_with_pipe = PickleableFunctionWithPipeIO(
                    func, child_conn, log_pipe.pipe.get_writer(), venv_executable
                )
                args_serialized = [cloudpickle.dumps(arg) for arg in args]
                kwargs_serialized = {
                    key: cloudpickle.dumps(value) for key, value in kwargs.items()
                }
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
