/**
 * \author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2022 Freek Stulp
 *
 * DmpBbo is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * DmpBbo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with DmpBbo.  If not, see <http://www.gnu.org/licenses/>.
 */

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "dynamicalsystems/DynamicalSystem.hpp"
#include "eigenutils/eigen_realtime_check.hpp"

using namespace std;
using namespace DmpBbo;
using namespace Eigen;
using namespace nlohmann;

int main(int n_args, char** args)
{
  // ./exec  => read json from default files, do not write output
  // ./exec <input json 1> .. <input json N>

  vector<string> filenames;

  if (n_args == 1) {
    // No directories and filename provided, add the files from the json
    // directory
    string input_directory = "../demos/cpp/json/";
    for (string system : {"Exponential", "Sigmoid", "SpringDamper"})
      for (string dim : {"1D", "2D"})
        filenames.push_back(input_directory + system + "System_" + dim +
                            "_for_cpp.json");
    filenames.push_back(input_directory + "TimeSystem_for_cpp.json");
    filenames.push_back(input_directory + "TimeCountDownSystem_for_cpp.json");

  } else {
    for (int i_args = 1; i_args < n_args; i_args++)
      filenames.push_back(args[i_args]);
  }

  for (string filename : filenames) {
    cout << "========================================================" << endl;
    cout << "Loading: " << filename << endl;

    cout << "===============" << endl;
    ifstream file(filename);
    if (file.fail()) {
      cerr << "ERROR: Could not find file: " << filename << endl;
      return -1;
    }
    json j = json::parse(file);
    cout << j << endl;

    cout << "===============" << endl;
    DynamicalSystem* d = j.get<DynamicalSystem*>();
    cout << *d << endl;

    // Prepare numerical integration
    VectorXd x(d->dim(), 1);
    VectorXd x_updated(d->dim(), 1);
    VectorXd xd(d->dim(), 1);
    double dt = 0.01;

    cout << "===============" << endl << "Integrating with Euler" << endl;
    d->integrateStart(x, xd);
    Eigen::internal::set_is_malloc_allowed(
        false);  // Make sure the following is real-time
    for (int t = 1; t < 10; t++) d->integrateStepEuler(dt, x, x_updated, xd);
    Eigen::internal::set_is_malloc_allowed(true);

    cout << "===============" << endl << "Integrating with Runge-Kutta" << endl;
    d->integrateStart(x, xd);
    Eigen::internal::set_is_malloc_allowed(
        false);  // Make sure the following is real-time
    for (int t = 1; t < 10; t++)
      d->integrateStepRungeKutta(dt, x, x_updated, xd);
    Eigen::internal::set_is_malloc_allowed(true);
  }

  return 0;
}
