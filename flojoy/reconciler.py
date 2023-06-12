"""
Nodes should try to accomodate any reasonable combination of inputs that a first-time Flojoy Studio user might try.

For example, the ADD node should make a best effort to do something reasonable when a matrix is added to a dataframe, or a 2 matrices of a different size are added.

For this reason, we've created the `Reconciler` class to handle the process of turning different data types into compatible, easily added objects. 
"""
from typing import Tuple
import numpy
import pandas

from .data_container import DataContainer


class IrreconcilableContainersException(Exception):
    pass


class Reconciler:
    def __init__(self, pad: float = 0):
        self.pad = pad

    def reconcile(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        types_to_reconcile = set([lhs.type, rhs.type])
        if types_to_reconcile == set(["matrix"]):
            return self.reconcile__matrix(lhs, rhs)
        elif types_to_reconcile == set(["dataframe"]):
            return self.reconcile__dataframe(lhs, rhs)
        elif types_to_reconcile == set(["ordered_pair"]):
            return self.reconcile__ordered_pair(lhs, rhs)
        elif types_to_reconcile == set(["matrix", "scalar"]):
            return self.reconcile__matrix_scalar(lhs, rhs)
        elif types_to_reconcile == set(["matrix", "dataframe"]):
            return self.reconcile__dataframe_matrix(lhs, rhs)
        elif types_to_reconcile == set(["scalar", "dataframe"]):
            return self.reconcile__dataframe_scalar(lhs, rhs)
        else:
            raise IrreconcilableContainersException(
                "FloJoy doesn't know how to reconcile data containers of type %s and %s"
                % (lhs.type, rhs.type)
            )

    def reconcile__matrix(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        # make the matrices equal sizes, by padding
        final_r = max(lhs.m.shape[0], rhs.m.shape[0])
        final_c = max(lhs.m.shape[1], rhs.m.shape[1])

        new_lhs = numpy.pad(
            lhs.m,
            ((0, final_r - lhs.m.shape[0]), (0, final_c - lhs.m.shape[1])),
            "constant",
            constant_values=self.pad,
        )
        new_rhs = numpy.pad(
            rhs.m,
            ((0, final_r - rhs.m.shape[0]), (0, final_c - rhs.m.shape[1])),
            "constant",
            constant_values=self.pad,
        )

        return (
            DataContainer(type="matrix", m=new_lhs),
            DataContainer(type="matrix", m=new_rhs),
        )

    def reconcile__dataframe(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        # pandas' handling for dataframes is actually pretty permissive. Let's just
        #  return both types as normal
        return (lhs, rhs)

    def reconcile__dataframe_scalar(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        # let's expand the scalar to be a dataframe the same size as the other dataframe
        if lhs.type == "dataframe":
            new_m = lhs.m.copy()
            new_m.iloc[:] = rhs.c
            return lhs, DataContainer(type="dataframe", m=new_m)

        new_m = rhs.m.copy()
        new_m.iloc[:] = lhs.c
        return DataContainer(type="dataframe", m=new_m), rhs

    def reconcile__ordered_pair(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        raise NotImplementedError("TODO")

    def reconcile__matrix_scalar(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        raise NotImplementedError("TODO")

    def reconcile__dataframe_matrix(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        raise NotImplementedError("TODO")
