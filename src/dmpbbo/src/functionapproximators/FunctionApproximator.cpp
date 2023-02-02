/**
 * @file   FunctionApproximator.cpp
 * @brief  FunctionApproximator class source file.
 * @author Freek Stulp
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

#include "functionapproximators/FunctionApproximator.hpp"

#include <eigen3/Eigen/Core>
#include <iostream>
#include <nlohmann/json.hpp>

#include "functionapproximators/FunctionApproximatorLWR.hpp"
#include "functionapproximators/FunctionApproximatorRBFN.hpp"

using namespace Eigen;
using namespace std;

namespace DmpBbo {

void from_json(const nlohmann::json& j, FunctionApproximator*& obj)
{
  string class_name = j.at("class").get<string>();

  if (class_name == "FunctionApproximatorRBFN") {
    obj = j.get<FunctionApproximatorRBFN*>();

  } else if (class_name == "FunctionApproximatorLWR") {
    obj = j.get<FunctionApproximatorLWR*>();

  } else {
    cerr << __FILE__ << ":" << __LINE__ << ":";
    cerr << "Unknown FunctionApproximator: " << class_name << endl;
    obj = NULL;
  }
}

std::ostream& operator<<(std::ostream& output,
                         const FunctionApproximator& function_approximator)
{
  nlohmann::json j;
  function_approximator.to_json_helper(j);
  // output << j.dump(4);
  output << j.dump();
  return output;
}

}  // namespace DmpBbo
