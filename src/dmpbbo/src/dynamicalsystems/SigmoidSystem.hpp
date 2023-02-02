/**
 * @file SigmoidSystem.hpp
 * @brief  SigmoidSystem class header file.
 * @author Freek Stulp
 *
 * This file is part of DmpBbo, a set of libraries and programs for the
 * black-box optimization of dynamical movement primitives.
 * Copyright (C) 2014 Freek Stulp, ENSTA-ParisTech
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

#ifndef _SIGMOID_SYSTEM_H_
#define _SIGMOID_SYSTEM_H_

#define EIGEN_RUNTIME_NO_MALLOC  // Enable runtime tests for allocations

#include <eigen3/Eigen/Core>
#include <nlohmann/json_fwd.hpp>

#include "dynamicalsystems/DynamicalSystem.hpp"

namespace DmpBbo {

/** \brief Dynamical System modelling the evolution of a sigmoidal system
 * \f$\dot{x} = -\alpha x(1-x/K)\f$.
 *
 * \ingroup DynamicalSystems
 */
class SigmoidSystem : public DynamicalSystem {
 public:
  /**
   *  Initialization constructor for a 1D system.
   *  \param tau              Time constant,                cf.
   * DynamicalSystem::tau() \param x_init           Initial state, cf.
   * DynamicalSystem::x_init() \param max_rate         Maximum rate of
   * change,       cf. SigmoidSystem::max_rate() \param inflection_ratio
   * Time at which maximum rate of change is achieved,  i.e. at inflection_ratio
   * * tau
   */
  SigmoidSystem(double tau, const Eigen::VectorXd& x_init, double max_rate,
                double inflection_ratio);

  /** Destructor. */
  ~SigmoidSystem(void);

  void differentialEquation(const Eigen::Ref<const Eigen::VectorXd>& x,
                            Eigen::Ref<Eigen::VectorXd> xd) const;

  void analyticalSolution(const Eigen::VectorXd& ts, Eigen::MatrixXd& xs,
                          Eigen::MatrixXd& xds) const;

  void set_tau(double tau);
  void set_x_init(const Eigen::VectorXd& x_init);

  /** Read an object from json.
   *  \param[in]  j   json input
   *  \param[out] obj The object read from json
   *
   * See also: https://github.com/nlohmann/json/issues/1324
   */
  friend void from_json(const nlohmann::json& j, SigmoidSystem*& obj);

  /** Write an object to json.
   *  \param[in] obj The object to write to json
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  inline friend void to_json(nlohmann::json& j, const SigmoidSystem* const& obj)
  {
    obj->to_json_helper(j);
  }

 private:
  /** Write this object to json.
   *  \param[out]  j json output
   *
   * See also:
   *   https://github.com/nlohmann/json/issues/1324
   *   https://github.com/nlohmann/json/issues/716
   */
  void to_json_helper(nlohmann::json& j) const;

  static Eigen::VectorXd computeKs(const Eigen::VectorXd& N_0s, double r,
                                   double inflection_point_time);

  double max_rate_;
  double inflection_ratio_;
  Eigen::VectorXd Ks_;
};

}  // namespace DmpBbo

#endif  // _Sigmoid_SYSTEM_H_
