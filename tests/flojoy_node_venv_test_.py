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
    def hello_world_jax():
        # Import jax to check if it is installed
        import jax
        # Fetch the list of installed packages
        import pkg_resources
        packages = pkg_resources.working_set
        packages_dict = {package.key: package.version for package in packages}
        return packages_dict
    
    # Run the function
    packages_dict = hello_world_jax()
    assert packages_dict["jax"] == "0.4.13"

#TODO(roulbac): Add more tests for
# 1 - Making sure the venv interpreter is separate from the parent process
# 2 - Making sure that the sys.path of the venv is appropriate
# 3 - Making sure that packages in the parent process are not available in the venv