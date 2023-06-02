"""
Nodes should try to accomodate any reasonable combination of inputs that a first-time Flojoy Studio user might try.

For example, the ADD node should make a best effort to do something reasonable when a matrix is added to a dataframe, or a 2 matrices of a different size are added.

For this reason, we've created the `Reconciler` class to handle the process of turning different data types into compatible, easily added objects. 
"""
from typing import Tuple
import numpy as np
import pandas as pd

from .data_container import DataContainer


class Reconciler:
    """ """

    def __init__(self):
        pass

    def reconcile(
        self, lhs: DataContainer, rhs: DataContainer
    ) -> Tuple[DataContainer, DataContainer]:
        return lhs, rhs
