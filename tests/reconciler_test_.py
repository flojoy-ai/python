import numpy
import unittest

from unittest.mock import patch


from flojoy.data_container import DataContainer
from flojoy.reconciler import Reconciler, IrreconcilableContainersException


class ReconcilerTestCase(unittest.TestCase):
    def test_matrix_different_sizes(self):
        # create the two ordered pair datacontainers
        dc_a = DataContainer(type="matrix", m=numpy.ones([2, 3]))

        dc_b = DataContainer(type="matrix", m=numpy.ones([3, 2]))

        r = Reconciler()
        # function under test
        rec_a, rec_b = r.reconcile(dc_a, dc_b)

        self.assertTrue(
            numpy.array_equal(
                numpy.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
                rec_a.m,
            )
        )
        self.assertTrue(
            numpy.array_equal(
                numpy.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
                rec_b.m,
            )
        )

    def test_complain_if_irreconcilable(self):
        dc_a = DataContainer(type="grayscale")

        dc_b = DataContainer(
            type="ordered_pair", x=numpy.linspace(-10, 10, 100), y=[7] * 100
        )

        r = Reconciler()
        # function under test
        with self.assertRaises(IrreconcilableContainersException):
            rec_a, rec_b = r.reconcile(dc_a, dc_b)
