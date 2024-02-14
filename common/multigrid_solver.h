/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2019 by Martin Kronbichler
 *
 * This project is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the multigrid project.
 *
 * ---------------------------------------------------------------------
 *
 * Inspired by the step-37 tutorial program of the deal.II finite element
 * library, www.dealii.org (LGPL license).
 *
 * This file implements the multigrid solver class implementing a V-cycle and
 * a full multigrid setup (as a solver) with user-supplied number of
 * cycles. It utilizes the laplace_operator.h file for the Laplace
 * operator. This function assumes the artificial case of a given analytic
 * solution that is based to this function and used for both the Dirichlet
 * boundary conditions and the evaluation of the L2 error.
 *
 * In the current design, it is assumed that all Dirichlet boundaries have the
 * id '0', whereas all other boundary ids correspond to Neumann boundaries.
 */

#ifndef multigrid_multigrid_solver_h
#define multigrid_multigrid_solver_h


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include "laplace_operator.h"


namespace multigrid
{
  using namespace dealii;


  namespace internal
  {
    template <typename Number, typename OtherNumber>
    void
    add_vector(LinearAlgebra::distributed::Vector<Number>            &dst,
               const LinearAlgebra::distributed::Vector<OtherNumber> &src)
    {
      AssertDimension(dst.locally_owned_size(), src.locally_owned_size());
      const OtherNumber *src_ptr    = src.begin();
      Number            *dst_ptr    = dst.begin();
      const unsigned int local_size = dst.locally_owned_size();

      DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < local_size; ++i)
        dst_ptr[i] += src_ptr[i];
    }
  } // namespace internal


  // A coarse solver defined via the smoother
  template <typename VectorType, typename SmootherType>
  class MGCoarseFromSmoother : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseFromSmoother(const SmootherType &mg_smoother, const bool is_empty)
      : smoother(mg_smoother)
      , is_empty(is_empty)
    {}

    virtual void
    operator()(const unsigned int level, VectorType &dst, const VectorType &src) const override
    {
      if (is_empty)
        return;
      smoother[level].vmult(dst, src);
    }

    const SmootherType &smoother;
    const bool          is_empty;
  };



  // Mixed-precision multigrid solver setup
  template <int dim, int fe_degree, typename Number, typename Number2>
  class MultigridSolver
  {
  public:
    MultigridSolver(const DoFHandler<dim> &dof_handler,
                    const Function<dim>   &boundary_values,
                    const Function<dim>   &right_hand_side,
                    const Function<dim>   &coefficient,
                    const unsigned int     degree_pre,
                    const unsigned int     degree_post,
                    const unsigned int     n_cycles = 1)
      : dof_handler(&dof_handler)
      , minlevel(0)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , residual(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , solution_update(minlevel, maxlevel)
      , matrix(minlevel, maxlevel)
      , matrix_dp(minlevel, maxlevel)
      , smooth(minlevel, maxlevel)
      , coarse(smooth, false)
      , degree_pre(degree_pre)
      , degree_post(degree_post)
      , n_cycles(n_cycles)
      , timings(maxlevel + 1)
      , analytic_solution(boundary_values)
    {
      Assert(degree_post == degree_pre,
             ExcNotImplemented("Change of pre- and post-smoother degree "
                               "currently not possible with deal.II"));

      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      // Initialization of Dirichlet boundaries
      std::set<types::boundary_id> dirichlet_boundary;
      dirichlet_boundary.insert(0);
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

      // set up a mapping for the geometry representation
      MappingQGeneric<dim> mapping(std::min(fe_degree, 10));

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
          AffineConstraints<double> level_constraints;
          level_constraints.reinit(relevant_dofs);
          level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
          level_constraints.close();

          // single-precision matrix-free data
          {
            typename MatrixFree<dim, Number>::AdditionalData additional_data;
            additional_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
            additional_data.mapping_update_flags =
              (update_gradients | update_JxW_values | update_quadrature_points);
            additional_data.mg_level = level;
            std::shared_ptr<MatrixFree<dim, Number>> mg_mf_storage_level(
              new MatrixFree<dim, Number>());
            mg_mf_storage_level->reinit(
              mapping, dof_handler, level_constraints, QGauss<1>(fe_degree + 1), additional_data);

            matrix[level].initialize(mg_mf_storage_level,
                                     level_constraints,
                                     mg_constrained_dofs,
                                     level);
            matrix[level].evaluate_coefficient(coefficient);

            matrix[level].initialize_dof_vector(defect[level]);
            t[level]               = defect[level];
            solution_update[level] = defect[level];
          }

          // double-precision matrix-free data
          {
            typename MatrixFree<dim, Number2>::AdditionalData additional_data;
            additional_data.tasks_parallel_scheme = MatrixFree<dim, Number2>::AdditionalData::none;
            additional_data.mapping_update_flags =
              (update_gradients | update_JxW_values | update_quadrature_points);
            additional_data.mg_level = level;
            AffineConstraints<double> unconstrained;
            unconstrained.close();
            std::vector<const AffineConstraints<double> *> constraints(
              {&level_constraints, &unconstrained});
            std::vector<const DoFHandler<dim> *> dof_handlers({&dof_handler, &dof_handler});
            std::vector<QGauss<1>>               quadratures;
            quadratures.emplace_back(fe_degree + 1);
            std::shared_ptr<MatrixFree<dim, Number2>> mg_mf_storage_level(
              new MatrixFree<dim, Number2>());
            mg_mf_storage_level->reinit(
              mapping, dof_handlers, constraints, quadratures, additional_data);

            matrix_dp[level].initialize(mg_mf_storage_level,
                                        level_constraints,
                                        mg_constrained_dofs,
                                        level);
            matrix_dp[level].evaluate_coefficient(coefficient);
            matrix_dp[level].initialize_dof_vector(solution[level]);
            rhs[level]      = solution[level];
            residual[level] = solution[level];
          }
        }

      Timer time;

      // build two level transfers; one is without boundary conditions for the
      // transfer of the solution (with inhomogeneous boundary conditions),
      // and one is for the homogeneous part in the v-cycle
      {
        std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners(
          dof_handler.get_triangulation().n_global_levels());
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          partitioners[level] = solution[level].get_partitioner();
        mg_transfer_no_boundary.build(dof_handler, partitioners);
      }
      {
        std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners(
          dof_handler.get_triangulation().n_global_levels());
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          partitioners[level] = solution_update[level].get_partitioner();
        transfer.initialize_constraints(mg_constrained_dofs);
        transfer.build(dof_handler, partitioners);
      }

      // interpolate the inhomogeneous boundary conditions
      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel + 1);
      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          Quadrature<dim - 1> face_quad(dof_handler.get_fe().get_unit_face_support_points());
          FEFaceValues<dim>   fe_values(mapping,
                                      dof_handler.get_fe(),
                                      face_quad,
                                      update_quadrature_points);
          std::vector<types::global_dof_index> face_dof_indices(dof_handler.get_fe().dofs_per_face);
          typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin(level),
                                                  endc = dof_handler.end(level);
          for (; cell != endc; ++cell)
            if (cell->level_subdomain_id() != numbers::artificial_subdomain_id)
              for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
                if (cell->at_boundary(face_no))
                  {
                    const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
                    face->get_mg_dof_indices(level, face_dof_indices);
                    fe_values.reinit(cell, face_no);
                    for (unsigned int i = 0; i < face_dof_indices.size(); ++i)
                      if (dof_handler.locally_owned_mg_dofs(level).is_element(face_dof_indices[i]))
                        {
                          const double value =
                            analytic_solution.value(fe_values.quadrature_point(i));
                          if (value != 0.0)
                            inhomogeneous_bc[level][face_dof_indices[i]] = value;
                        }
                  }

          // evaluate the right hand side in the equation, including the
          // residual from the inhomogeneous boundary conditions
          for (auto &i : inhomogeneous_bc[level])
            if (dof_handler.locally_owned_mg_dofs(level).is_element(i.first))
              solution[level](i.first) = i.second;

          matrix_dp[level].compute_residual(rhs[level], solution[level], right_hand_side);
        }
      const double rhs_norm = rhs[maxlevel].l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time compute rhs:      " << time.wall_time() << " rhs_norm = " << rhs_norm
                  << std::endl;

      time.restart();
      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          typename SmootherType::AdditionalData smoother_data;
          if (level > minlevel)
            {
              smoother_data.smoothing_range     = 20.;
              smoother_data.degree              = degree_pre;
              smoother_data.eig_cg_n_iterations = 15;
              smoother_data.polynomial_type =
                SmootherType::AdditionalData::PolynomialType::first_kind;
            }
          else
            {
              smoother_data.smoothing_range     = 1e-3;
              smoother_data.degree              = numbers::invalid_unsigned_int;
              smoother_data.eig_cg_n_iterations = matrix[minlevel].m();
            }
          matrix[level].compute_diagonal();
          smoother_data.preconditioner = matrix[level].get_matrix_diagonal_inverse();
          smooth[level].initialize(matrix[level], smoother_data);
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time initial smoother: " << time.wall_time() << std::endl;
    }



    // Compute the L2 error of the 'solution' field on a given level, weighted
    // by the volume of the domain
    double
    compute_l2_error(const unsigned int level)
    {
      for (auto &i : inhomogeneous_bc[level])
        solution[level](i.first) = i.second;
      solution[level].update_ghost_values();

      double                                                  global_error  = 0;
      double                                                  global_volume = 0;
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number2> phi(
        *matrix_dp[level].get_matrix_free(), 0, 0);
      for (unsigned int cell = 0; cell < matrix_dp[level].get_matrix_free()->n_cell_batches();
           ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values_plain(solution[level]);
          phi.evaluate(EvaluationFlags::values);
          VectorizedArray<Number2> local_error  = VectorizedArray<Number2>();
          VectorizedArray<Number2> local_volume = VectorizedArray<Number2>();
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              VectorizedArray<Number2> exact_values;
              auto                     p_vec = phi.quadrature_point(q);
              for (unsigned int v = 0; v < VectorizedArray<Number2>::size(); ++v)
                {
                  Point<dim> p;
                  for (unsigned int d = 0; d < dim; ++d)
                    p[d] = p_vec[d][v];
                  exact_values[v] = analytic_solution.value(p);
                }
              local_error +=
                (phi.get_value(q) - exact_values) * (phi.get_value(q) - exact_values) * phi.JxW(q);
              local_volume += phi.JxW(q);
            }
          for (unsigned int v = 0;
               v < matrix_dp[level].get_matrix_free()->n_active_entries_per_cell_batch(cell);
               ++v)
            {
              global_error += local_error[v];
              global_volume += local_volume[v];
            }
        }
      global_error  = Utilities::MPI::sum(global_error, MPI_COMM_WORLD);
      global_volume = Utilities::MPI::sum(global_volume, MPI_COMM_WORLD);
      return std::sqrt(global_error / global_volume);
    }



    // Print a summary of computation times on the various levels
    void
    print_wall_times()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "Coarse solver " << (int)timings[minlevel][1]
                    << " times: " << timings[minlevel][0] << " tot prec " << timings[minlevel][2]
                    << std::endl;
          std::cout << "level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC"
                    << std::endl;
          for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
            {
              std::cout << "L" << std::setw(2) << std::left << level << "    ";
              std::cout << std::setprecision(4) << std::setw(12) << timings[level][5]
                        << std::setw(10) << timings[level][0] << std::setw(10) << timings[level][4]
                        << std::setw(10) << timings[level][1] << std::setw(12) << timings[level][2]
                        << std::setw(10) << timings[level][3] << std::endl;
            }
          std::cout << std::setprecision(5);
        }
      for (unsigned int l = 0; l < timings.size(); ++l)
        for (unsigned int j = 0; j < timings[l].size(); ++j)
          timings[l][j] = 0.;
    }



    // Return the solution vector for further processing
    const LinearAlgebra::distributed::Vector<Number2> &
    get_solution()
    {
      for (auto &i : inhomogeneous_bc[maxlevel])
        solution[maxlevel](i.first) = i.second;
      return solution[maxlevel];
    }



    // Solve with the FMG cycle and return the reduction rate of a V-cycle
    double
    solve(const bool do_analyze)
    {
      double reduction_rate = 1.;

      Timer time;

      // copy double to float, invoke coarse solver twice (improves accuracy
      // for high-order methods where 1e-3 might not be enough, and this is
      // done only once anyway), and copy back to double
      defect[minlevel] = rhs[minlevel];
      coarse(minlevel, t[minlevel], defect[minlevel]);
      smooth[minlevel].step(t[minlevel], defect[minlevel]);
      solution[minlevel] = t[minlevel];
      timings[minlevel][0] += time.wall_time();
      timings[minlevel][1] += 2;

      for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
        {
          // interpolate inhomogeneous boundary values
          Timer time;
          for (auto &i : inhomogeneous_bc[level - 1])
            solution[level - 1](i.first) = i.second;
          timings[level][3] += time.wall_time();

          // prolongate (without boundary conditions) to next finer level in
          // double precision
          time.restart();
          mg_transfer_no_boundary.prolongate(level, solution[level], solution[level - 1]);
          timings[level][2] += time.wall_time();

          double init_residual = 1.;

          if (do_analyze)
            {
              const double l2_error = compute_l2_error(level);
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "error start         level " << level << ": " << l2_error << std::endl;
            }

          for (auto &i : inhomogeneous_bc[level])
            solution[level](i.first) = 0;

          // compute residual in double precision
          time.restart();
          matrix_dp[level].vmult_residual(rhs[level], solution[level], residual[level]);
          timings[level][0] += time.wall_time();

          time.restart();
          // copy to single precision
          defect[level] = residual[level];

          timings[level][4] += time.wall_time();


          if (do_analyze)
            {
              const double res_norm = residual[level].l2_norm();
              init_residual         = res_norm;
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "residual norm start level " << level << ": " << res_norm << std::endl;
            }

          // run v-cycle to obtain correction
          v_cycle(level, n_cycles);

          time.restart();

          // add correction
          internal::add_vector(solution[level], solution_update[level]);

          timings[level][4] += time.wall_time();

          if (do_analyze)
            {
              for (auto &i : inhomogeneous_bc[level])
                solution[level](i.first) = 0;
              matrix_dp[level].vmult(residual[level], solution[level]);
              residual[level].sadd(-1., 1., rhs[level]);
              const double res_norm = residual[level].l2_norm();
              reduction_rate        = std::pow(res_norm / init_residual, 1. / n_cycles);
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "residual norm end   level " << level << ": " << res_norm << std::endl;
              const double l2_error = compute_l2_error(level);
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "error end           level " << level << ": " << l2_error << std::endl;
            }
        }
      return reduction_rate;
    }



    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult() or vmult_with_residual_update()) and return the
    // number of iterations and the reduction rate per CG iteration
    std::pair<unsigned int, double>
    solve_cg()
    {
      ReductionControl      solver_control(100, 1e-16, 1e-9);
      SolverCG<VectorType2> solver_cg(solver_control);
      solution[maxlevel] = 0;
      solver_cg.solve(matrix_dp[maxlevel], solution[maxlevel], rhs[maxlevel], *this);
      return std::make_pair(solver_control.last_step(),
                            std::pow(solver_control.last_value() / solver_control.initial_value(),
                                     1. / solver_control.last_step()));
    }



    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number2>       &dst,
          const LinearAlgebra::distributed::Vector<Number2> &src) const
    {
      Timer time1, time;
      defect[maxlevel].copy_locally_owned_data_from(src);
      timings[maxlevel][4] += time.wall_time();
      v_cycle(maxlevel, 1);
      time.restart();
      dst.copy_locally_owned_data_from(solution_update[maxlevel]);
      timings[maxlevel][4] += time.wall_time();
      timings[minlevel][2] += time1.wall_time();
    }



    // Implement the vmult_with_residual_update() function ensuring that the
    // CG solver switches to the fast path with merged vector operations
    std::array<Number2, 2>
    vmult_with_residual_update(LinearAlgebra::distributed::Vector<Number2> &residual,
                               LinearAlgebra::distributed::Vector<Number2> &update,
                               const Number2                                factor) const
    {
      Timer time1, time;
      AssertDimension(residual.locally_owned_size(), update.locally_owned_size());
      AssertDimension(defect[maxlevel].locally_owned_size(), residual.locally_owned_size());

      const unsigned int local_size   = matrix[maxlevel].local_size_without_constraints();
      Number2           *update_ptr   = update.begin();
      Number2           *residual_ptr = residual.begin();
      Number            *defect_ptr   = defect[maxlevel].begin();
      if (factor != Number2())
        DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < local_size; ++i)
        defect_ptr[i] = residual_ptr[i] + factor * update_ptr[i];
      else DEAL_II_OPENMP_SIMD_PRAGMA for (unsigned int i = 0; i < local_size; ++i) defect_ptr[i] =
        residual_ptr[i];

      timings[maxlevel][4] += time.wall_time();

      v_cycle(maxlevel, 1);

      time.restart();
      const Number            *solution_update_ptr = solution_update[maxlevel].begin();
      VectorizedArray<Number2> inner_product = {}, inner_product2 = {};
      constexpr unsigned int   n_lanes     = VectorizedArray<Number2>::size();
      const unsigned int       regular_end = local_size / n_lanes * n_lanes;
      if (factor != Number2())
        {
          for (unsigned int i = 0; i < regular_end; i += n_lanes)
            {
              VectorizedArray<Number2> mg_result, upd, res;
              DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int v = 0; v < n_lanes; ++v)
                mg_result[v] = solution_update_ptr[i + v];
              upd.load(update_ptr + i);
              res.load(residual_ptr + i);
              res += upd * factor;
              inner_product += mg_result * res;
              inner_product2 += mg_result * upd * factor;
              res.store(residual_ptr + i);
              mg_result.store(update_ptr + i);
            }
          for (unsigned int i = regular_end; i < local_size; ++i)
            {
              Number2 mg_result = solution_update_ptr[i];
              residual_ptr[i] += update_ptr[i] * factor;
              inner_product[0] += mg_result * residual_ptr[i];
              inner_product2[0] += mg_result * update_ptr[i] * factor;
              update_ptr[i] = mg_result;
            }

          // fix constrained dofs according to 1 on diagonal
          for (unsigned int i = local_size; i < residual.local_size(); ++i)
            {
              residual_ptr[i] += factor * update_ptr[i];
              inner_product[0] += residual_ptr[i] * residual_ptr[i];
              inner_product2[0] += residual_ptr[i] * factor * update_ptr[i];
              update_ptr[i] = residual_ptr[i];
            }
        }
      else
        {
          for (unsigned int i = 0; i < regular_end; i += n_lanes)
            {
              VectorizedArray<Number2> mg_result, old_residual;
              DEAL_II_OPENMP_SIMD_PRAGMA
              for (unsigned int v = 0; v < n_lanes; ++v)
                mg_result[v] = solution_update_ptr[i + v];
              old_residual.load(residual_ptr + i);
              inner_product += mg_result * old_residual;
              mg_result.store(update_ptr + i);
            }
          for (unsigned int i = regular_end; i < local_size; ++i)
            {
              Number2 mg_result = solution_update_ptr[i];
              inner_product[0] += mg_result * residual_ptr[i];
              update_ptr[i] = mg_result;
            }

          // fix constrained dofs according to 1 on diagonal
          for (unsigned int i = local_size; i < residual.locally_owned_size(); ++i)
            {
              inner_product[0] += residual_ptr[i] * residual_ptr[i];
              update_ptr[i] = residual_ptr[i];
            }
          inner_product2 = inner_product;
        }
      for (unsigned int v = 1; v < n_lanes; ++v)
        inner_product[0] += inner_product[v];
      for (unsigned int v = 1; v < n_lanes; ++v)
        inner_product2[0] += inner_product2[v];

      std::array<Number2, 2> results({inner_product[0], inner_product2[0]});
      Utilities::MPI::sum(ArrayView<const Number2>(results.data(), 2),
                          residual.get_mpi_communicator(),
                          ArrayView<Number2>(results.data(), 2));

      timings[maxlevel][4] += time.wall_time();
      timings[minlevel][2] += time1.wall_time();
      return results;
    }



    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix_dp[maxlevel].vmult(residual[maxlevel], solution[maxlevel]);
    }



    // run matrix-vector product in single precision
    void
    do_matvec_smoother()
    {
      matrix[maxlevel].vmult(solution_update[maxlevel], defect[maxlevel]);
    }

  private:
    // Implement the V-cycle
    void
    v_cycle(const unsigned int level, const unsigned int my_n_cycles) const
    {
      if (level == minlevel)
        {
          Timer time;
          (coarse)(level, solution_update[level], defect[level]);
          timings[level][0] += time.wall_time();
          timings[level][1] += 1;
          return;
        }

      for (unsigned int c = 0; c < my_n_cycles; ++c)
        {
          Timer time;
          if (c == 0)
            (smooth)[level].vmult(solution_update[level], defect[level]);
          else
            (smooth)[level].step(solution_update[level], defect[level]);
          timings[level][5] += time.wall_time();

          time.restart();
          (matrix)[level].vmult_residual(defect[level], solution_update[level], t[level]);
          timings[level][0] += time.wall_time();

          time.restart();
          defect[level - 1] = 0;
          transfer.restrict_and_add(level, defect[level - 1], t[level]);
          timings[level][1] += time.wall_time();

          v_cycle(level - 1, 1);

          time.restart();
          transfer.prolongate_and_add(level, solution_update[level], solution_update[level - 1]);
          timings[level][2] += time.wall_time();

          time.restart();
          (smooth)[level].step(solution_update[level], defect[level]);
          timings[level][5] += time.wall_time();
        }
    }


    const SmartPointer<const DoFHandler<dim>> dof_handler;

    std::vector<std::map<types::global_dof_index, Number2>> inhomogeneous_bc;

    MGConstrainedDoFs mg_constrained_dofs;

    MGTransferMatrixFree<dim, Number2> mg_transfer_no_boundary;
    MGTransferMatrixFree<dim, Number>  transfer;

    typedef LinearAlgebra::distributed::Vector<Number>  VectorType;
    typedef LinearAlgebra::distributed::Vector<Number2> VectorType2;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * The solution vector
     */
    mutable MGLevelObject<VectorType2> solution;

    /**
     * Original right hand side vector
     */
    mutable MGLevelObject<VectorType2> rhs;

    /**
     * Residual vector before it is passed down into float through the v-cycle
     */
    mutable MGLevelObject<VectorType2> residual;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType> t;

    /**
     * Auxiliary vector for the solution update
     */
    mutable MGLevelObject<VectorType> solution_update;

    /**
     * The matrix for each level
     */
    MGLevelObject<LaplaceOperator<dim, fe_degree, Number>> matrix;

    /**
     * The double-precision matrix for the outer correction
     */
    MGLevelObject<LaplaceOperator<dim, fe_degree, Number2>> matrix_dp;

    /**
     * The smoother object
     */
    typedef PreconditionChebyshev<LaplaceOperator<dim, fe_degree, Number>, VectorType> SmootherType;
    MGLevelObject<SmootherType>                                                        smooth;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorType, MGLevelObject<SmootherType>> coarse;

    /**
     * Chebyshev degree for pre-smoothing
     */
    const unsigned int degree_pre;

    /**
     * Chebyshev degree for post-smoothing
     */
    const unsigned int degree_post;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<std::array<double, 6>> timings;

    /**
     * Function for boundary values that we keep as analytic solution
     */
    const Function<dim> &analytic_solution;
  };



  // Double-precision multigrid solver setup, implemented as a specialization
  // of the other templated class. Quite some code could be shared, but not
  // all because we do not need as many vectors.
  template <int dim, int fe_degree, typename Number>
  class MultigridSolver<dim, fe_degree, Number, Number>
  {
  public:
    MultigridSolver(const DoFHandler<dim> &dof_handler,
                    const Function<dim>   &boundary_values,
                    const Function<dim>   &right_hand_side,
                    const Function<dim>   &coefficient,
                    const unsigned int     degree_pre,
                    const unsigned int     degree_post,
                    const unsigned int     n_cycles = 1)
      : dof_handler(&dof_handler)
      , minlevel(0)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , matrix(minlevel, maxlevel)
      , smooth(minlevel, maxlevel)
      , coarse(smooth, false)
      , degree_pre(degree_pre)
      , degree_post(degree_post)
      , n_cycles(n_cycles)
      , timings(maxlevel + 1)
      , analytic_solution(boundary_values)
    {
      Assert(degree_post == degree_pre,
             ExcNotImplemented("Change of pre- and post-smoother degree "
                               "currently not possible with deal.II"));

      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      // Initialization of Dirichlet boundaries
      std::set<types::boundary_id> dirichlet_boundary;
      dirichlet_boundary.insert(0);
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

      // set up a mapping for the geometry representation
      MappingQGeneric<dim> mapping(std::min(fe_degree, 10));

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
          AffineConstraints<double> level_constraints;
          level_constraints.reinit(relevant_dofs);
          level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
          level_constraints.close();

          // matrix-free data field
          typename MatrixFree<dim, Number>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
          additional_data.mapping_update_flags =
            (update_gradients | update_JxW_values | update_quadrature_points);
          additional_data.mg_level = level;
          std::vector<const AffineConstraints<double> *> constraints(1, &level_constraints);
          std::vector<const DoFHandler<dim> *>           dof_handlers(1, &dof_handler);
          std::vector<QGauss<1>>                         quadratures;
          quadratures.emplace_back(fe_degree + 1);
          std::shared_ptr<MatrixFree<dim, Number>> mg_mf_storage_level(
            new MatrixFree<dim, Number>());
          mg_mf_storage_level->reinit(
            mapping, dof_handlers, constraints, quadratures, additional_data);

          matrix[level].initialize(mg_mf_storage_level, constraints, mg_constrained_dofs, level);
          matrix[level].evaluate_coefficient(coefficient);

          matrix[level].initialize_dof_vector(solution[level]);
          defect[level] = solution[level];
          rhs[level]    = solution[level];
          t[level]      = solution[level];
        }

      Timer time;

      // build two level transfers; one is without boundary conditions for the
      // transfer of the solution (with inhomogeneous boundary conditions),
      // and one is for the homogeneous part in the v-cycle
      mg_transfer_no_boundary.build(dof_handler);
      transfer.initialize_constraints(mg_constrained_dofs);
      transfer.build(dof_handler);

      // interpolate the inhomogeneous boundary conditions
      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel + 1);
      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          Quadrature<dim - 1> face_quad(dof_handler.get_fe().get_unit_face_support_points());
          FEFaceValues<dim>   fe_values(mapping,
                                      dof_handler.get_fe(),
                                      face_quad,
                                      update_quadrature_points);
          std::vector<types::global_dof_index> face_dof_indices(dof_handler.get_fe().dofs_per_face);

          typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin(level),
                                                  endc = dof_handler.end(level);
          for (; cell != endc; ++cell)
            if (cell->level_subdomain_id() != numbers::artificial_subdomain_id)
              for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell; ++face_no)
                if (cell->at_boundary(face_no))
                  {
                    const typename DoFHandler<dim>::face_iterator face = cell->face(face_no);
                    face->get_mg_dof_indices(level, face_dof_indices);
                    fe_values.reinit(cell, face_no);
                    for (unsigned int i = 0; i < face_dof_indices.size(); ++i)
                      if (dof_handler.locally_owned_mg_dofs(level).is_element(face_dof_indices[i]))
                        {
                          const double value =
                            analytic_solution.value(fe_values.quadrature_point(i));
                          if (value != 0.0)
                            inhomogeneous_bc[level][face_dof_indices[i]] = value;
                        }
                  }

          // evaluate the right hand side in the equation, including the
          // residual from the inhomogeneous boundary conditions
          for (auto &i : inhomogeneous_bc[level])
            if (dof_handler.locally_owned_mg_dofs(level).is_element(i.first))
              solution[level](i.first) = i.second;

          solution[level].update_ghost_values();
          FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(
            *matrix[level].get_matrix_free());
          for (unsigned int cell = 0; cell < matrix[level].get_matrix_free()->n_cell_batches();
               ++cell)
            {
              phi.reinit(cell);
              phi.read_dof_values_plain(solution[level]);
              phi.evaluate(EvaluationFlags::gradients);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                {
                  Point<dim, VectorizedArray<Number>> pvec = phi.quadrature_point(q);
                  VectorizedArray<Number>             rhs_val;
                  for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                    {
                      Point<dim> p;
                      for (unsigned int d = 0; d < dim; ++d)
                        p[d] = pvec[d][v];
                      rhs_val[v] = right_hand_side.value(p);
                    }
                  phi.submit_value(rhs_val, q);
                  phi.submit_gradient(-phi.get_gradient(q), q);
                }
              phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients,
                                    rhs[level]);
            }
          rhs[level].compress(VectorOperation::add);
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time compute rhs:      " << time.wall_time() << std::endl;

      time.restart();
      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          typename SmootherType::AdditionalData smoother_data;
          if (level > minlevel)
            {
              smoother_data.smoothing_range     = 20.;
              smoother_data.degree              = degree_pre;
              smoother_data.eig_cg_n_iterations = 15;
              smoother_data.polynomial_type =
                SmootherType::AdditionalData::PolynomialType::fourth_kind;
            }
          else
            {
              smoother_data.smoothing_range     = 1e-3;
              smoother_data.degree              = numbers::invalid_unsigned_int;
              smoother_data.eig_cg_n_iterations = matrix[minlevel].m();
            }
          matrix[level].compute_diagonal();
          smoother_data.preconditioner = matrix[level].get_matrix_diagonal_inverse();
          smooth[level].initialize(matrix[level], smoother_data);
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time initial smoother: " << time.wall_time() << std::endl;
    }

    // Compute the L2 error of the 'solution' field on a given level, weighted
    // by the volume of the domain
    double
    compute_l2_error(const unsigned int level) const
    {
      for (auto &i : inhomogeneous_bc[level])
        solution[level](i.first) = i.second;
      solution[level].update_ghost_values();

      double                                                 global_error  = 0;
      double                                                 global_volume = 0;
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(*matrix[level].get_matrix_free(),
                                                                 0,
                                                                 0);
      for (unsigned int cell = 0; cell < matrix[level].get_matrix_free()->n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values_plain(solution[level]);
          phi.evaluate(EvaluationFlags::values);
          VectorizedArray<Number> local_error  = VectorizedArray<Number>();
          VectorizedArray<Number> local_volume = VectorizedArray<Number>();
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            {
              VectorizedArray<Number> exact_values;
              auto                    p_vec = phi.quadrature_point(q);
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  Point<dim> p;
                  for (unsigned int d = 0; d < dim; ++d)
                    p[d] = p_vec[d][v];
                  exact_values[v] = analytic_solution.value(p);
                }
              local_error +=
                (phi.get_value(q) - exact_values) * (phi.get_value(q) - exact_values) * phi.JxW(q);
              local_volume += phi.JxW(q);
            }
          for (unsigned int v = 0;
               v < matrix[level].get_matrix_free()->n_active_entries_per_cell_batch(cell);
               ++v)
            {
              global_error += local_error[v];
              global_volume += local_volume[v];
            }
        }
      global_error  = Utilities::MPI::sum(global_error, MPI_COMM_WORLD);
      global_volume = Utilities::MPI::sum(global_volume, MPI_COMM_WORLD);
      return std::sqrt(global_error / global_volume);
    }


    // Print a summary of computation times on the various levels
    void
    print_wall_times()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "Coarse solver " << (int)timings[minlevel][1]
                    << " times: " << timings[minlevel][0] << " tot prec " << timings[minlevel][2]
                    << std::endl;
          std::cout << "level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC"
                    << std::endl;
          for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
            {
              std::cout << "L" << std::setw(2) << std::left << level << "    ";
              std::cout << std::setprecision(4) << std::setw(12) << timings[level][5]
                        << std::setw(10) << timings[level][0] << std::setw(10) << timings[level][4]
                        << std::setw(10) << timings[level][1] << std::setw(12) << timings[level][2]
                        << std::setw(10) << timings[level][3] << std::endl;
            }
          std::cout << std::setprecision(5);
          for (unsigned int l = 0; l < timings.size(); ++l)
            for (unsigned int j = 0; j < timings[l].size(); ++j)
              timings[l][j] = 0.;
        }
    }

    const LinearAlgebra::distributed::Vector<Number> &
    get_solution()
    {
      for (auto &i : inhomogeneous_bc[maxlevel])
        solution[maxlevel](i.first) = i.second;
      return solution[maxlevel];
    }

    // Solve with the FMG cycle and return the reduction rate of a V-cycle
    double
    solve(const bool do_analyze) const
    {
      double reduction_rate = 1.;

      Timer time;
      coarse(minlevel, solution[minlevel], rhs[minlevel]);
      smooth[minlevel].step(solution[minlevel], rhs[minlevel]);
      timings[minlevel][0] += time.wall_time();
      timings[minlevel][1] += 1;

      for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
        {
          Timer time;
          for (auto &i : inhomogeneous_bc[level - 1])
            solution[level - 1](i.first) = i.second;
          timings[level][3] += time.wall_time();

          double init_residual = 1.;

          time.restart();
          mg_transfer_no_boundary.prolongate(level, solution[level], solution[level - 1]);
          timings[level][2] += time.wall_time();

          if (do_analyze)
            {
              for (auto &i : inhomogeneous_bc[level])
                solution[level](i.first) = 0;
              matrix[level].vmult(t[level], solution[level]);
              t[level].sadd(-1., 1., rhs[level]);
              init_residual = t[level].l2_norm();
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "residual norm start level " << level << ": " << init_residual
                          << std::endl;
              const double l2_error = compute_l2_error(level);
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "error start         level " << level << ": " << l2_error << std::endl;
              for (auto &i : inhomogeneous_bc[level])
                solution[level](i.first) = 0;
            }

          time.restart();
          defect[level] = rhs[level];
          timings[level][4] += time.wall_time();

          v_cycle(level, true);

          if (do_analyze)
            {
              for (auto &i : inhomogeneous_bc[level])
                solution[level](i.first) = 0;
              matrix[level].vmult(t[level], solution[level]);
              t[level].sadd(-1., 1., rhs[level]);
              const double res_norm = t[level].l2_norm();
              reduction_rate        = std::pow(res_norm / init_residual, 1. / n_cycles);
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "residual norm end   level " << level << ": " << res_norm << std::endl;
              const double l2_error = compute_l2_error(level);
              if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
                std::cout << "error end           level " << level << ": " << l2_error << std::endl;
            }
        }
      return reduction_rate;
    }

    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult) and return the number of iterations and the
    // reduction rate per CG iteration
    std::pair<unsigned int, double>
    solve_cg()
    {
      ReductionControl                           solver_control(100, 1e-16, 1e-9);
      SolverCG<VectorType>                       solver_cg(solver_control);
      LinearAlgebra::distributed::Vector<Number> solution_update = solution[maxlevel];
      solution_update                                            = 0;
      solver_cg.solve(matrix[maxlevel], solution_update, rhs[maxlevel], *this);
      solution[maxlevel] = solution_update;
      return std::make_pair(solver_control.last_step(),
                            std::pow(solver_control.last_value() / solver_control.initial_value(),
                                     1. / solver_control.last_step()));
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      Timer time;
      transfer.copy_to_mg(*dof_handler, defect, src);
      v_cycle(maxlevel, false);
      transfer.copy_from_mg(*dof_handler, dst, solution);
    }

    void
    do_matvec()
    {
      matrix[maxlevel].vmult(t[maxlevel], solution[maxlevel]);
    }

    void
    do_matvec_smoother()
    {
      matrix[maxlevel].vmult(t[maxlevel], solution[maxlevel]);
    }

  private:
    /**
     * Implements the v-cycle
     */
    void
    v_cycle(const unsigned int level, const bool outer_solution) const
    {
      if (level == minlevel)
        {
          Timer time;
          (coarse)(level, solution[level], defect[level]);
          timings[level][0] += time.wall_time();
          return;
        }

      for (unsigned int c = 0; c < (outer_solution ? n_cycles : 1); ++c)
        {
          Timer time;
          if (outer_solution == false && c == 0)
            (smooth)[level].vmult(solution[level], defect[level]);
          else
            (smooth)[level].step(solution[level], defect[level]);
          timings[level][5] += time.wall_time();

          time.restart();
          (matrix)[level].vmult(t[level], solution[level]);
          t[level].sadd(-1.0, 1.0, defect[level]);
          timings[level][0] += time.wall_time();

          time.restart();
          defect[level - 1] = 0;
          transfer.restrict_and_add(level, defect[level - 1], t[level]);
          timings[level][1] += time.wall_time();

          v_cycle(level - 1, false);

          time.restart();
          transfer.prolongate_and_add(level, solution[level], solution[level - 1]);
          timings[level][2] += time.wall_time();

          time.restart();
          (smooth)[level].step(solution[level], defect[level]);
          timings[level][5] += time.wall_time();
        }
    }

    const SmartPointer<const DoFHandler<dim>> dof_handler;

    std::vector<std::map<types::global_dof_index, Number>> inhomogeneous_bc;

    MGConstrainedDoFs mg_constrained_dofs;

    MGTransferMatrixFree<dim, Number> mg_transfer_no_boundary;
    MGTransferMatrixFree<dim, Number> transfer;

    typedef LinearAlgebra::distributed::Vector<Number> VectorType;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * The solution update after the multigrid step.
     */
    mutable MGLevelObject<VectorType> solution;

    /**
     * Right hand side vector
     */
    mutable MGLevelObject<VectorType> rhs;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType> t;

    /**
     * The matrix for each level.
     */
    MGLevelObject<LaplaceOperator<dim, fe_degree, Number>> matrix;

    /**
     * The smoother object.
     */
    typedef PreconditionChebyshev<LaplaceOperator<dim, fe_degree, Number>, VectorType> SmootherType;
    MGLevelObject<SmootherType>                                                        smooth;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorType, MGLevelObject<SmootherType>> coarse;

    /**
     * Chebyshev degree for pre-smoothing
     */
    const unsigned int degree_pre;

    /**
     * Chebyshev degree for post-smoothing
     */
    const unsigned int degree_post;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<std::array<double, 6>> timings;

    /**
     * Function for boundary values that we keep as analytic solution
     */
    const Function<dim> &analytic_solution;
  };
} // namespace multigrid

#endif
