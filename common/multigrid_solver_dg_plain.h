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
 * This file implements the multigrid solver class implementing a V-cycle
 * within a DG solver with transfer to continuous finite elements and further
 * h-coarsening. It utilizes the DG discretization from laplace_operator_dg.h
 * as well as the file laplace_operator.h for the continuous FEM Laplace
 * operator. This function assumes the artificial case of a given analytic
 * solution.
 *
 * In the current design, it is assumed that all Dirichlet boundaries have the
 * id '0', whereas all other boundary ids correspond to Neumann boundaries.
 */

#ifndef multigrid_multigrid_solver_dg_plain_h
#define multigrid_multigrid_solver_dg_plain_h


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include "laplace_operator.h"
#include "laplace_operator_dg.h"
#include "multigrid_solver.h"


namespace multigrid
{
  using namespace dealii;


  // Mixed-precision multigrid solver setup
  template <int dim, int fe_degree, typename Number, typename Number2, int type = 0>
  class MultigridSolverDGPlain
  {
  public:
    MultigridSolverDGPlain(const DoFHandler<dim> &dof_handler,
                           const Function<dim>   &boundary_values,
                           const Function<dim>   &right_hand_side,
                           const Function<dim> & /*coefficient*/,
                           const unsigned int degree_pre,
                           const unsigned int degree_post,
                           const unsigned int n_cycles = 1)
      : dof_handler(&dof_handler)
      , minlevel(0)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , solution_update(minlevel, maxlevel)
      , matrix(minlevel, maxlevel)
      , smooth(minlevel, maxlevel)
      , coarse(smooth, false)
      , degree_pre(degree_pre)
      , degree_post(degree_post)
      , n_cycles(n_cycles)
      , timings(maxlevel + 2)
      , analytic_solution(boundary_values)
    {
      Assert(degree_post == degree_pre,
             ExcNotImplemented("Change of pre- and post-smoother degree "
                               "currently not possible with deal.II"));

      AssertDimension(fe_degree, dof_handler.get_fe().degree);
      Timer time;

      // set up a mapping for the geometry representation
      MappingQGeneric<dim> mapping(std::min(fe_degree, 10));

      typename MatrixFree<dim, Number>::AdditionalData mf_data;
      mf_data.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::none;
      mf_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          AffineConstraints<double> level_constraints;
          level_constraints.close();

          // single-precision matrix-free data
          {
            std::shared_ptr<MatrixFree<dim, Number>> mg_mf_storage_level(
              new MatrixFree<dim, Number>());
            if (level < maxlevel)
              {
                mf_data.mg_level = level;
                mg_mf_storage_level->reinit(
                  mapping, dof_handler, level_constraints, QGauss<1>(fe_degree + 1), mf_data);
              }
            else
              {
                mf_data.mg_level = numbers::invalid_unsigned_int;
                mg_mf_storage_level->reinit(
                  mapping, dof_handler, level_constraints, QGauss<1>(fe_degree + 1), mf_data);
              }
            matrix[level].reinit(mg_mf_storage_level, 0);

            matrix[level].initialize_dof_vector(defect[level]);
            matrix[level].initialize_dof_vector(t[level]);
            matrix[level].initialize_dof_vector(solution_update[level]);
          }
        }
      print_time(time.wall_time(), "Time matrix-free SP", MPI_COMM_WORLD);
      time.restart();

      // double-precision matrix-free data
      {
        typename MatrixFree<dim, Number2>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, Number2>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_gradients | update_JxW_values | update_quadrature_points);
        additional_data.mg_level = numbers::invalid_unsigned_int;
        AffineConstraints<double> unconstrained;
        unconstrained.close();
        std::shared_ptr<MatrixFree<dim, Number2>> mg_mf_storage(new MatrixFree<dim, Number2>());
        mg_mf_storage->reinit(mapping,
                              dof_handler,
                              AffineConstraints<double>(),
                              QGauss<1>(fe_degree + 1),
                              additional_data);

        matrix_dg_dp.reinit(mg_mf_storage, 0);
        matrix_dg_dp.initialize_dof_vector(solution);
        rhs      = solution;
        residual = solution;
      }
      print_time(time.wall_time(), "Time matrix-free DP", MPI_COMM_WORLD);
      time.restart();

      // build two level transfers; one is without boundary conditions for the
      // transfer of the solution (with inhomogeneous boundary conditions),
      // and one is for the homogeneous part in the v-cycle
      {
        std::vector<std::shared_ptr<const Utilities::MPI::Partitioner>> partitioners(
          dof_handler.get_triangulation().n_global_levels());
        for (unsigned int level = minlevel; level <= maxlevel; ++level)
          partitioners[level] = solution_update[level].get_partitioner();
        transfer.build(dof_handler, partitioners);
      }
      print_time(time.wall_time(), "Time build MG transfer", MPI_COMM_WORLD);
      time.restart();

      {
        FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number2> phi(matrix_dg_dp.get_matrix_free(),
                                                                    0);
        for (unsigned int cell = 0; cell < matrix_dg_dp.get_matrix_free().n_cell_batches(); ++cell)
          {
            phi.reinit(cell);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              {
                Point<dim, VectorizedArray<Number2>> pvec = phi.quadrature_point(q);
                VectorizedArray<Number2>             rhs_val;
                for (unsigned int v = 0; v < VectorizedArray<Number2>::size(); ++v)
                  {
                    Point<dim> p;
                    for (unsigned int d = 0; d < dim; ++d)
                      p[d] = pvec[d][v];
                    rhs_val[v] = right_hand_side.value(p);
                  }
                phi.submit_value(rhs_val, q);
              }
            phi.integrate(EvaluationFlags::values);
            phi.set_dof_values(rhs);
          }
      }
      const double rhs_norm = rhs.l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Time compute rhs              " << time.wall_time()
                  << " rhs_norm = " << rhs_norm << std::endl;

      time.restart();
      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          typename SmootherType::AdditionalData smoother_data;
          if (level > minlevel)
            {
              smoother_data.smoothing_range = 20.;
              if (level < maxlevel)
                smoother_data.degree = degree_pre;
              else
                smoother_data.degree = std::max(1U, degree_pre - 1U);
              smoother_data.eig_cg_n_iterations = 15;
            }
          else
            {
              smoother_data.smoothing_range     = 1e-5;
              smoother_data.degree              = numbers::invalid_unsigned_int;
              smoother_data.eig_cg_n_iterations = matrix[minlevel].m();
            }
          smoother_data.preconditioner.reset(
            new JacobiTransformed<dim, fe_degree, Number, type>(matrix[level]));
          smooth[level].initialize(matrix[level], smoother_data);
        }
      print_time(time.wall_time(), "Time initial smoother", MPI_COMM_WORLD);
    }



    // Compute the L2 error of the 'solution' field on a given level, weighted
    // by the volume of the domain
    double
    compute_l2_error()
    {
      double                                                  global_error  = 0;
      double                                                  global_volume = 0;
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number2> phi(matrix_dg_dp.get_matrix_free());
      for (unsigned int cell = 0; cell < matrix_dg_dp.get_matrix_free().n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          phi.gather_evaluate(solution, EvaluationFlags::values);
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
               v < matrix_dg_dp.get_matrix_free().n_active_entries_per_cell_batch(cell);
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
      return solution;
    }



    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult() or vmult_with_residual_update()) and return the
    // number of iterations and the reduction rate per CG iteration
    std::pair<double, double>
    solve_cg(const double tolerance)
    {
      ReductionControl      solver_control(100, 1e-16, tolerance);
      SolverCG<VectorType2> solver_cg(solver_control);
      solution = 0;
      solver_cg.solve(matrix_dg_dp,
                      solution,
                      rhs, // PreconditionIdentity());
                      *this);
      const double reduction_rate =
        std::pow(solver_control.last_value() / solver_control.initial_value(),
                 1. / solver_control.last_step());
      return std::make_pair(std::log(tolerance) / std::log(reduction_rate), reduction_rate);
    }



    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number2>       &dst,
          const LinearAlgebra::distributed::Vector<Number2> &src) const
    {
      Timer time1, time;
      defect[maxlevel].copy_locally_owned_data_from(src);
      timings[maxlevel + 1][4] += time.wall_time();
      v_cycle(maxlevel, 1);
      time.restart();
      dst.copy_locally_owned_data_from(solution_update[maxlevel]);
      timings[maxlevel + 1][4] += time.wall_time();
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
      AssertDimension(residual.local_size(), update.local_size());
      AssertDimension(defect[maxlevel].local_size(), residual.local_size());

      const unsigned int local_size   = residual.local_size();
      Number2           *update_ptr   = update.begin();
      Number2           *residual_ptr = residual.begin();
      Number            *defect_ptr   = defect[maxlevel].begin();
      if (factor != Number2())
        DEAL_II_OPENMP_SIMD_PRAGMA
      for (unsigned int i = 0; i < local_size; ++i)
        defect_ptr[i] = residual_ptr[i] + factor * update_ptr[i];
      else DEAL_II_OPENMP_SIMD_PRAGMA for (unsigned int i = 0; i < local_size; ++i) defect_ptr[i] =
        residual_ptr[i];

      timings[maxlevel + 1][4] += time.wall_time();

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

      timings[maxlevel + 1][4] += time.wall_time();
      timings[minlevel][2] += time1.wall_time();
      return results;
    }



    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix_dg_dp.vmult(residual, solution);
    }



    // run matrix-vector product in single precision
    void
    do_matvec_smoother()
    {
      matrix[maxlevel].vmult(solution_update[maxlevel], defect[maxlevel]);
    }

    void
    print_matvec_details()
    {
      matrix_dg_dp.print_and_reset_wall_times();
      matrix[maxlevel].print_and_reset_wall_times();
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

    MGTransferMatrixFree<dim, Number> transfer;

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
    mutable VectorType2 solution;

    /**
     * Original right hand side vector
     */
    mutable VectorType2 rhs;

    /**
     * Residual vector before it is passed down into float through the v-cycle
     */
    mutable VectorType2 residual;

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
    MGLevelObject<LaplaceOperatorCompactCombine<dim, fe_degree, Number, type>> matrix;
    LaplaceOperatorCompactCombine<dim, fe_degree, Number2, type>               matrix_dg_dp;

    /**
     * The smoother object
     */
    typedef PreconditionChebyshev<LaplaceOperatorCompactCombine<dim, fe_degree, Number, type>,
                                  VectorType,
                                  JacobiTransformed<dim, fe_degree, Number, type>>
                                SmootherType;
    MGLevelObject<SmootherType> smooth;

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
