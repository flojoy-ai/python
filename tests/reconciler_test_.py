import numpy

from unittest.mock import patch

from flojoy.data_container import DataContainer
from flojoy.reconciler import Reconciler


def test_initialize():
    # create the two ordered pair datacontainers
    dc_a = DataContainer(
        type="ordered_pair", x=numpy.linspace(-10, 10, 100), y=[10] * 100
    )

    dc_b = DataContainer(
        type="ordered_pair", x=numpy.linspace(-10, 10, 100), y=[7] * 100
    )

    r = Reconciler()
    rec_a, rec_b = r.reconcile(dc_a, dc_b)

    assert rec_a is not None
