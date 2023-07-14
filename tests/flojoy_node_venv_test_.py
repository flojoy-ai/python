import pytest
import os
import shutil
from unittest.mock import patch

from flojoy import flojoy, run_in_venv


# Define a fixture to patch tempfile.tempdir
@pytest.fixture
def mock_tempdir():
    _test_tempdir = "/tmp/run_in_venv_tests"
    # Wipe the directory to be patched if it exists
    shutil.rmtree(_test_tempdir, ignore_errors=True)
    os.makedirs(_test_tempdir)
    # Patch the tempfile.tempdir
    with patch("tempfile.tempdir", _test_tempdir):
        yield _test_tempdir
    # Clean up
    shutil.rmtree(_test_tempdir)


def test_run_in_venv_imports_properly(mock_tempdir):
    """Test that run_in_venv imports properly a package specified in its pip_dependencies argument"""

    @run_in_venv(pip_dependencies=["jax[cpu]==0.4.13"])
    def empty_function_with_jax():
        # Import jax to check if it is installed
        # Fetch the list of installed packages
        import sys
        import importlib.metadata
        import jax

        # Get the list of installed packages
        packages_dict = {package.name: package.version for package in importlib.metadata.distributions()}
        return packages_dict, sys.path, sys.executable

    # Run the function
    packages_dict, sys_path, sys_executable = empty_function_with_jax()
    print("EXECUTABLE: ", sys_executable)
    print("SYS_PATH: ", sys_path)
    print("PACKAGES_DICT: ", packages_dict)
    # # Test for executable
    # assert sys_executable.startswith(mock_tempdir)
    # # Test for sys.path
    # assert sys_path[-1].startswith(mock_tempdir)
    # # Test for package version
    # assert packages_dict["jax"] == "0.4.13"