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

import pdb



class FunctionApproximatorCustom(FunctionApproximator):

    def __init__(self, alpha = 4, num_bases=10, kernel="wendland8"):

        meta_params = {"num_bases": num_bases, "kernel": kernel, "alpha": alpha}

        # self.gen_centers()
        # self.gen_width()

        model_param_names = ["w"]


        super().__init__(meta_params, model_param_names)

    @staticmethod
    def _train(inputs, targets, meta_params, **kwargs):

        num_bases = meta_params["num_bases"]
        kernel = meta_params["kernel"]
        alpha = meta_params["alpha"]
        
        # Calculate the centers of the basis functions
        centers = np.exp(- alpha  * ((np.cumsum(np.ones([1, num_bases + 1])) - 1) / num_bases))

        # Calculate the "widths" for the basis functions
        if (kernel == 'gaussian'):
            width = 1.0 / np.diff(centers) / np.diff(centers)
            width = np.append(width, width[-1])
        else:
            width = 1.0 / np.diff(centers)
            width = np.append(width[0], width)
        
        # psi_set = FunctionApproximatorCustom.gen_psi(inputs, kernel, centers, width, num_bases)
        
        # s = inputs.squeeze()
        # # Generate the Psi
        # c = np.reshape(centers, [num_bases + 1, 1])
        # w = np.reshape(width, [num_bases + 1,1 ])
        # if (kernel == 'gaussian'):
        #     xi = w * (s - c) * (s - c)
        #     psi_set = np.exp(- xi)
        # else:
        #     xi = np.abs(w * (s - c))
        #     if (kernel == 'mollifier'):
        #         psi_set = (np.exp(- 1.0 / (1.0 - xi * xi))) * (xi < 1.0)
        #     elif (kernel == 'wendland2'):
        #         psi_set = ((1.0 - xi) ** 2.0) * (xi < 1.0)
        #     elif (kernel == 'wendland3'):
        #         psi_set = ((1.0 - xi) ** 3.0) * (xi < 1.0)
        #     elif (kernel == 'wendland4'):
        #         psi_set = ((1.0 - xi) ** 4.0 * (4.0 * xi + 1.0)) * (xi < 1.0)
        #     elif (kernel == 'wendland5'):
        #         psi_set = ((1.0 - xi) ** 5.0 * (5.0 * xi + 1)) * (xi < 1.0)
        #     elif (kernel == 'wendland6'):
        #         psi_set = ((1.0 - xi) ** 6.0 * 
        #             (35.0 * xi ** 2.0 + 18.0 * xi + 3.0)) * (xi < 1.0)
        #     elif (kernel == 'wendland7'):
        #         psi_set = ((1.0 - xi) ** 7.0 *
        #             (16.0 * xi ** 2.0 + 7.0 * xi + 1.0)) * (xi < 1.0)
        #     elif (kernel == 'wendland8'):
        #         psi_set = (((1.0 - xi) ** 8.0 *
        #             (32.0 * xi ** 3.0 + 25.0 * xi ** 2.0 + 8.0 * xi + 1.0)) *
        #             (xi < 1.0))
        # psi_set = np.nan_to_num(psi_set)        

        # sum_psi = np.sum(psi_set, 0)
        # P = psi_set / sum_psi * s
        # weights = np.nan_to_num(targets @ np.linalg.pinv(P))

        weights = FunctionApproximatorCustom.gen_weights(inputs, targets, kernel, 
                                                         centers, width, num_bases)
        print(weights.shape)
        model_params = {"weights": weights, "centers": centers, "width": width, 
                        "kernel": kernel, "num_bases": num_bases}

        return model_params

    @staticmethod
    def _predict(inputs, model_params):
        weights = model_params["weights"]
        centers = model_params["centers"]
        width = model_params["width"]
        kernel = model_params["kernel"]
        num_bases = model_params["num_bases"]
        psi_set = FunctionApproximatorCustom.gen_psi(inputs, kernel, centers, width, num_bases)
        f = np.dot(weights,psi_set[:,0]) / np.sum(psi_set[:,0]) * inputs
        # pdb.set_trace()
        return f

    def plot_model_parameters(self, inputs_min, inputs_max, **kwargs):
        """ Plot a representation of the model parameters on a grid.

        @param inputs_min: The min values for the grid
        @param inputs_max:  The max values for the grid
        @return:
        """
        ax = kwargs.get("ax") or self._get_axis()
        # No model parameters to plot
        return [], ax
    
    @staticmethod
    def gen_psi(inputs, kernel, centers, width, num_bases):
        s = inputs.squeeze()
        # Generate the Psi
        c = np.reshape(centers, [num_bases + 1, 1])
        w = np.reshape(width, [num_bases + 1,1 ])
        if (kernel == 'gaussian'):
            xi = w * (s - c) * (s - c)
            psi_set = np.exp(- xi)
        else:
            xi = np.abs(w * (s - c))
            if (kernel == 'mollifier'):
                psi_set = (np.exp(- 1.0 / (1.0 - xi * xi))) * (xi < 1.0)
            elif (kernel == 'wendland2'):
                psi_set = ((1.0 - xi) ** 2.0) * (xi < 1.0)
            elif (kernel == 'wendland3'):
                psi_set = ((1.0 - xi) ** 3.0) * (xi < 1.0)
            elif (kernel == 'wendland4'):
                psi_set = ((1.0 - xi) ** 4.0 * (4.0 * xi + 1.0)) * (xi < 1.0)
            elif (kernel == 'wendland5'):
                psi_set = ((1.0 - xi) ** 5.0 * (5.0 * xi + 1)) * (xi < 1.0)
            elif (kernel == 'wendland6'):
                psi_set = ((1.0 - xi) ** 6.0 * 
                    (35.0 * xi ** 2.0 + 18.0 * xi + 3.0)) * (xi < 1.0)
            elif (kernel == 'wendland7'):
                psi_set = ((1.0 - xi) ** 7.0 *
                    (16.0 * xi ** 2.0 + 7.0 * xi + 1.0)) * (xi < 1.0)
            elif (kernel == 'wendland8'):
                psi_set = (((1.0 - xi) ** 8.0 *
                    (32.0 * xi ** 3.0 + 25.0 * xi ** 2.0 + 8.0 * xi + 1.0)) *
                    (xi < 1.0))
        psi_set = np.nan_to_num(psi_set)
        # print(psi_set.shape)
        return psi_set
    
    @staticmethod
    def gen_weights(inputs, targets, kernel, centers, width, num_bases):
        psi_set = FunctionApproximatorCustom.gen_psi(inputs, kernel, centers, width, num_bases)
        s = inputs.squeeze()
        sum_psi = np.sum(psi_set, 0)
        P = psi_set / sum_psi * s
        weights = np.nan_to_num(targets @ np.linalg.pinv(P))
        return weights