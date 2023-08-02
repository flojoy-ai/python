import pytest
import os
import shutil
from unittest.mock import patch
import tempfile


pytestmark = pytest.mark.slow


# Define a fixture to patch tempfile.tempdir
@pytest.fixture
def mock_venv_cache_dir():
    _test_tempdir = os.path.join(tempfile.gettempdir(), "test_flojoy_node_venv")
    # Wipe the directory to be patched if it exists
    shutil.rmtree(_test_tempdir, ignore_errors=True)
    os.makedirs(_test_tempdir)
    # Patch the tempfile.tempdir
    with patch(
        "flojoy.flojoy_node_venv._get_venv_cache_dir", return_value=_test_tempdir
    ):
        yield _test_tempdir
    # Clean up
    shutil.rmtree(_test_tempdir)


def test_run_in_venv_imports_jax_properly(mock_venv_cache_dir):
    """Test that run_in_venv imports properly jax for example"""

    from flojoy import flojoy, run_in_venv

    @run_in_venv(pip_dependencies=["jax[cpu]==0.4.13"])
    def empty_function_with_jax():
        # Import jax to check if it is installed
        # Fetch the list of installed packages
        import sys
        import importlib.metadata
        import jax

        # Get the list of installed packages
        packages_dict = {
            package.name: package.version
            for package in importlib.metadata.distributions()
        }
        return packages_dict, sys.path, sys.executable

    # Run the function
    packages_dict, sys_path, sys_executable = empty_function_with_jax()
    # Test for executable
    assert sys_executable.startswith(mock_venv_cache_dir)
    # Test for sys.path
    assert sys_path[-1].startswith(mock_venv_cache_dir)
    # Test for package version
    assert packages_dict["jax"] == "0.4.13"


def test_run_in_venv_imports_flytekit_properly(mock_venv_cache_dir):
    from flojoy import flojoy, run_in_venv

    # Define a function that imports flytekit and returns its version
    @run_in_venv(pip_dependencies=["flytekit==1.8.2"])
    def empty_function_with_flytekit():
        import sys
        import importlib.metadata
        import flytekit

        # Get the list of installed packages
        packages_dict = {
            package.name: package.version
            for package in importlib.metadata.distributions()
        }
        return packages_dict, sys.path, sys.executable

    # Run the function
    packages_dict, sys_path, sys_executable = empty_function_with_flytekit()
    # Test for executable
    assert sys_executable.startswith(mock_venv_cache_dir)
    # Test for sys.path
    assert sys_path[-1].startswith(mock_venv_cache_dir)
    # Test for package version
    assert packages_dict["flytekit"] == "1.8.2"


def test_run_in_venv_imports_opencv_properly(mock_venv_cache_dir):
    # Define a function that imports opencv-python-headless and returns its version

    from flojoy import flojoy, run_in_venv

    @run_in_venv(pip_dependencies=["opencv-python-headless==4.7.0.72"])
    def empty_function_with_opencv():
        import sys
        import importlib.metadata
        import cv2

        # Get the list of installed packages
        packages_dict = {
            package.name: package.version
            for package in importlib.metadata.distributions()
        }
        return packages_dict, sys.path, sys.executable

    # Run the function
    packages_dict, sys_path, sys_executable = empty_function_with_opencv()
    # Test for executable
    assert sys_executable.startswith(mock_venv_cache_dir)
    # Test for sys.path
    assert sys_path[-1].startswith(mock_venv_cache_dir)
    # Test for package version
    assert packages_dict["opencv-python-headless"] == "4.7.0.72"


def test_run_in_venv_does_not_hang_on_error(mock_venv_cache_dir):
    """Test that run_in_venv imports properly jax for example"""

    from flojoy import run_in_venv

    @run_in_venv(pip_dependencies=[])
    def empty_function_with_error():
        return 1 / 0

    # Run the function and expect an error
    with pytest.raises(ChildProcessError):
        empty_function_with_error()
