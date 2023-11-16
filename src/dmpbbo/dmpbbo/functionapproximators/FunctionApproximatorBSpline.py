# This file is part of DmpBbo, a set of libraries and programs for the
# black-box optimization of dynamical movement primitives.
# Copyright (C) 2018, 2022 Freek Stulp
#
# DmpBbo is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# DmpBbo is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
""" Module for the FunctionApproximatorWLS class. """

import numpy as np
from scipy import interpolate

from dmpbbo.functionapproximators.FunctionApproximator import FunctionApproximator


class FunctionApproximatorBSpline(FunctionApproximator):
    """ A BSpline  function approximator.
    """

    def __init__(self, k=3):

        meta_params = {"k": k}

        model_param_names = ["knots", "coeffs", "k"]


        super().__init__(meta_params, model_param_names)

    @staticmethod
    def _train(inputs, targets, meta_params, **kwargs):

        k = meta_params["k"]

        x = inputs[::-1]
        y = targets[::-1]

        tck = interpolate.splrep(x, y, k=k)

        model_params = {"knots": tck[0], "coeffs": tck[1], "k": tck[2]}

        return model_params

    @staticmethod
    def _predict(inputs, model_params):

        x = inputs[::-1]

        tck = (model_params["knots"], model_params["coeffs"], model_params["k"])

        outputs = interpolate.BSpline(*tck)(x)


        return outputs[::-1]

    def plot_model_parameters(self, inputs_min, inputs_max, **kwargs):
        """ Plot a representation of the model parameters on a grid.

        @param inputs_min: The min values for the grid
        @param inputs_max:  The max values for the grid
        @return:
        """
        ax = kwargs.get("ax") or self._get_axis()
        # No model parameters to plot
        return [], ax
