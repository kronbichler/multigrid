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
 * This program solves the minimal surface equation on a circle/ball, a
 * nonlinear variant of the Laplace equation. The nonlinearity is resolved
 * with Newton's method and a line search procedure for globalization. The
 * linear system is solved with the conjugate gradient method preconditioned
 * by a geometric multigrid V-cycle.
 */


#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "../common/laplace_operator.h"


namespace multigrid
{
  using namespace dealii;


  const unsigned int degree_finite_element = 4;
  const unsigned int dimension             = 2;
  const bool         use_jacobi            = false;

  typedef double number;
  typedef float  level_number;

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution()
      : Function<dim>()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;
  };



  template <int dim>
  double
  Solution<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    return std::sin(2 * numbers::PI * (p[0] + p[1]));
  }



  template <int dim, int fe_degree, typename number>
  class MinimalSurfaceOperator : public LaplaceOperator<dim, fe_degree, number>
  {
  public:
    typedef number value_type;

    void
    compute_residual(LinearAlgebra::distributed::Vector<number> &      dst,
                     const LinearAlgebra::distributed::Vector<number> &src,
                     bool                                              first_time = false) const;

    void
    evaluate_coefficient(const bool                                  first_time,
                         LinearAlgebra::distributed::Vector<number> &solution);
  };



  template <int dim, int fe_degree, typename number>
  void
  MinimalSurfaceOperator<dim, fe_degree, number>::evaluate_coefficient(
    const bool                                  first_time,
    LinearAlgebra::distributed::Vector<number> &solution)
  {
    this->merged_coefficient.resize(this->data->n_cell_batches() * this->data->get_n_q_points());
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> fe_eval(*this->data, 1);
    solution.update_ghost_values();
    for (unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell)
      {
        const std::size_t data_ptr =
          this->data->get_mapping_info().cell_data[0].data_index_offsets[cell];
        fe_eval.reinit(cell);
        fe_eval.gather_evaluate(solution, false, true);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            Tensor<2, dim, VectorizedArray<number>> tensor =
              first_time ?
                unit_symmetric_tensor<dim, VectorizedArray<number>>() :
                (unit_symmetric_tensor<dim, VectorizedArray<number>>() -
                 symmetrize(outer_product(fe_eval.get_gradient(q),
                                          (1. / (1. + fe_eval.get_gradient(q).norm_square()) *
                                           fe_eval.get_gradient(q))))) /
                  std::sqrt(1. + fe_eval.get_gradient(q).norm_square());

            const unsigned int stride =
              this->data->get_mapping_info().get_cell_type(cell) < 2 ? 0 : 1;
            Tensor<2, dim, VectorizedArray<number>> coef =
              (this->data->get_mapping_info().get_cell_type(cell) < 2 ?
                 this->data->get_mapping_info().cell_data[0].descriptor[0].quadrature_weights[q] :
                 number(1.)) *
              this->data->get_mapping_info().cell_data[0].JxW_values[data_ptr + q * stride] *
              transpose(
                this->data->get_mapping_info().cell_data[0].jacobians[0][data_ptr + q * stride]) *
              tensor *
              this->data->get_mapping_info().cell_data[0].jacobians[0][data_ptr + q * stride];
            for (unsigned int d = 0; d < dim; ++d)
              this->merged_coefficient[cell * fe_eval.n_q_points + q][d] = coef[d][d];
            for (unsigned int c = 0, d = 0; d < dim; ++d)
              for (unsigned int e = d + 1; e < dim; ++e, ++c)
                this->merged_coefficient[cell * fe_eval.n_q_points + q][dim + c] = coef[d][e];
          }
      }
    solution.zero_out_ghosts();
  }



  template <int dim, int fe_degree, typename number>
  void
  MinimalSurfaceOperator<dim, fe_degree, number>::compute_residual(
    LinearAlgebra::distributed::Vector<number> &      dst,
    const LinearAlgebra::distributed::Vector<number> &src,
    bool                                              first_time) const
  {
    dst = 0;
    src.update_ghost_values();
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi(*this->data, 0);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, number> phi_nodirichlet(*this->data, 1);

    for (unsigned int cell = 0; cell < this->data->n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi_nodirichlet.reinit(cell);
        phi_nodirichlet.read_dof_values(src);
        phi_nodirichlet.evaluate(false, true);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_gradient(first_time ?
                                -phi_nodirichlet.get_gradient(q) :
                                -phi_nodirichlet.get_gradient(q) /
                                  std::sqrt(1. + phi_nodirichlet.get_gradient(q).norm_square()),
                              q);
        phi.integrate(false, true);
        phi.distribute_local_to_global(dst);
      }
    dst.compress(VectorOperation::add);
    src.zero_out_ghosts();
  }



  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();
    void
    run();

  private:
    void
    setup_system();
    void
    solve(const bool first_time);
    void
    output_results(const unsigned int cycle) const;
    void
    interpolate_boundary_values();

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    MappingQGeneric<dim> mapping;
    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;

    AffineConstraints<double>                constraints_without_dirichlet;
    AffineConstraints<double>                constraints;
    std::shared_ptr<MatrixFree<dim, number>> system_matrix_free;
    typedef MinimalSurfaceOperator<dim, degree_finite_element, number> SystemMatrixType;
    SystemMatrixType                                                   system_matrix;

    MGConstrainedDoFs                                                        mg_constrained_dofs;
    MGTransferMatrixFree<dim, level_number>                                  mg_transfer;
    typedef MinimalSurfaceOperator<dim, degree_finite_element, level_number> LevelMatrixType;
    MGLevelObject<LevelMatrixType>                                           mg_matrices;

    LinearAlgebra::distributed::Vector<number> solution;
    LinearAlgebra::distributed::Vector<number> search_direction;
    LinearAlgebra::distributed::Vector<number> tentative_solution;
    LinearAlgebra::distributed::Vector<number> system_rhs;

    double             setup_time;
    ConditionalOStream pcout;
    ConditionalOStream time_details;

    double       time_residual, time_solve;
    unsigned int n_residual, linear_iterations;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem()
    :
#ifdef DEAL_II_WITH_P4EST
    triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy)
    ,
#else
    triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
    ,
#endif
    mapping(degree_finite_element)
    , fe(degree_finite_element)
    , dof_handler(triangulation)
    , mg_transfer(mg_constrained_dofs)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_details(std::cout, false && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , time_residual(0)
    , time_solve(0)
    , n_residual(0)
    , linear_iterations(0)
  {}



  template <int dim>
  void
  LaplaceProblem<dim>::setup_system()
  {
    Timer time;
    time.start();
    setup_time = 0;

    system_matrix_free.reset(new MatrixFree<dim, number>());
    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    pcout << "Number of active cells:       " << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);
    constraints.close();
    constraints_without_dirichlet.clear();
    constraints_without_dirichlet.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints_without_dirichlet);
    constraints_without_dirichlet.close();
    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.                " << time.wall_time() << " s"
                 << std::endl;
    time.restart();

    std::vector<const DoFHandler<dim> *> dof_handlers(2, &dof_handler);
    {
      std::vector<const AffineConstraints<double> *> constraint(2);
      constraint[0] = &constraints;
      constraint[1] = &constraints_without_dirichlet;
      typename MatrixFree<dim, number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme = MatrixFree<dim, number>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);
      system_matrix_free->reinit(
        mapping, dof_handlers, constraint, QGauss<1>(fe.degree + 1), additional_data);
      std::vector<unsigned int> mask({0});
      system_matrix.initialize(system_matrix_free, constraints, mask);
    }
    system_matrix_free->initialize_dof_vector(solution, 1);
    system_matrix_free->initialize_dof_vector(search_direction, 0);
    system_matrix_free->initialize_dof_vector(tentative_solution, 1);
    system_matrix_free->initialize_dof_vector(system_rhs, 0);

    // VectorTools::interpolate(mapping, dof_handler, Solution<dim>(), solution);

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system              " << time.wall_time() << " s"
                 << std::endl;
    time.restart();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels - 1);

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

    for (unsigned int level = 0; level < nlevels; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();
        std::vector<const AffineConstraints<double> *> constraint(2);
        AffineConstraints<double>                      dummy;
        dummy.close();
        constraint[0] = &level_constraints;
        constraint[1] = &dummy;

        typename MatrixFree<dim, level_number>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme = MatrixFree<dim, level_number>::AdditionalData::none;
        additional_data.mapping_update_flags =
          (update_gradients | update_JxW_values | update_quadrature_points);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim, level_number>> mg_matrix_free_level(
          new MatrixFree<dim, level_number>());
        mg_matrix_free_level->reinit(
          mapping, dof_handlers, constraint, QGauss<1>(fe.degree + 1), additional_data);
        mg_matrices[level].initialize(mg_matrix_free_level, level_constraints,
                                      mg_constrained_dofs, level);
      }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels              " << time.wall_time() << " s"
                 << std::endl;

    time.restart();
    mg_transfer.build(dof_handler);
    setup_time += time.wall_time();
    time_details << "MG build transfer time                " << time.wall_time() << " s\n";
    pcout << "Total setup time               (wall) " << setup_time << " s\n";
    std::cout.precision(12);
  }



  template <int dim>
  void
  LaplaceProblem<dim>::interpolate_boundary_values()
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(
      mapping, dof_handler, 0, Solution<dim>(), boundary_values);
    for (typename std::map<types::global_dof_index, double>::iterator it = boundary_values.begin();
         it != boundary_values.end();
         ++it)
      if (solution.locally_owned_elements().is_element(it->first))
        solution(it->first) = it->second;

    constraints_without_dirichlet.distribute(solution);
  }



  template <int dim>
  void
  LaplaceProblem<dim>::solve(const bool first_time)
  {
    Timer                                                         time;
    std::vector<LinearAlgebra::distributed::Vector<level_number>> coefficient_solutions(
      triangulation.n_global_levels());
    coefficient_solutions.back() = solution;
    for (unsigned int level = triangulation.n_global_levels() - 1; level > 0; --level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
        LinearAlgebra::distributed::Vector<level_number> ghosted_vector(
          dof_handler.locally_owned_mg_dofs(level), relevant_dofs, MPI_COMM_WORLD);
        ghosted_vector = coefficient_solutions[level];
        ghosted_vector.update_ghost_values();
        mg_matrices[level - 1].initialize_dof_vector(coefficient_solutions[level - 1]);
        std::vector<level_number>                     dof_values_coarse(fe.dofs_per_cell);
        Vector<level_number>                          dof_values_fine(fe.dofs_per_cell);
        Vector<level_number>                          tmp(fe.dofs_per_cell);
        std::vector<types::global_dof_index>          dof_indices(fe.dofs_per_cell);
        typename DoFHandler<dim>::level_cell_iterator cell = dof_handler.begin_mg(level - 1);
        typename DoFHandler<dim>::level_cell_iterator endc = dof_handler.end_mg(level - 1);
        for (; cell != endc; ++cell)
          if (cell->is_locally_owned_on_level())
            {
              Assert(cell->has_children(), ExcNotImplemented());
              std::fill(dof_values_coarse.begin(), dof_values_coarse.end(), 0.);
              for (unsigned int child = 0; child < cell->n_children(); ++child)
                {
                  cell->child(child)->get_active_or_mg_dof_indices(dof_indices);
                  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                    dof_values_fine(i) = ghosted_vector(dof_indices[i]);
                  fe.get_restriction_matrix(child, cell->refinement_case())
                    .vmult(tmp, dof_values_fine);
                  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                    if (fe.restriction_is_additive(i))
                      dof_values_coarse[i] += tmp[i];
                    else if (tmp[i] != 0.)
                      dof_values_coarse[i] = tmp[i];
                }
              cell->get_active_or_mg_dof_indices(dof_indices);
              for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
                coefficient_solutions[level - 1](dof_indices[i]) = dof_values_coarse[i];
            }
        coefficient_solutions[level - 1].compress(VectorOperation::insert);
      }
    system_matrix.evaluate_coefficient(first_time, solution);

    typedef PreconditionChebyshev<LevelMatrixType, LinearAlgebra::distributed::Vector<level_number>>
      SmootherType;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<level_number>>
                                                         mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
      {
        mg_matrices[level].evaluate_coefficient(first_time, coefficient_solutions[level]);
        if (level > 0)
          {
            smoother_data[level].smoothing_range = 20.;
            if (use_jacobi)
              smoother_data[level].degree = 0;
            else
              smoother_data[level].degree = 2;
            smoother_data[level].eig_cg_n_iterations = 15;
          }
        else
          {
            smoother_data[0].smoothing_range     = 1e-3;
            smoother_data[0].degree              = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        mg_matrices[level].compute_diagonal();
        // pcout << coefficient_solutions[level].l2_norm() << " " <<
        // mg_matrices[level].get_matrix_diagonal_inverse()->get_vector().l2_norm() << std::endl;
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<level_number>> mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<level_number>> mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType>> mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels() - 1);
    for (unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<level_number>> mg_interface(
      mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<level_number>> mg(
      mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    // mg.set_debug(3);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<level_number>,
                   MGTransferMatrixFree<dim, level_number>>
      preconditioner(dof_handler, mg, mg_transfer);


    ReductionControl solver_control(system_matrix.m(), 1e-13, 1e-4);
    SolverCG<LinearAlgebra::distributed::Vector<number>> cg(solver_control);
    pcout << "MG build smoother time                " << time.wall_time() << " s\n";

    time.restart();
    system_matrix.compute_residual(system_rhs, solution, first_time);
    const double initial_residual_norm = system_rhs.l2_norm();
    time_residual += time.wall_time();
    ++n_residual;

    time.restart();

    // std::vector<LinearAlgebra::distributed::Vector<level_number> >
    // vectors(triangulation.n_global_levels()); vectors.back() = system_rhs; pcout <<
    // vectors.back().l2_norm() << " "; for (unsigned int level=triangulation.n_global_levels()-1;
    // level > 0; --level)
    // {
    //    mg_matrices[level-1].initialize_dof_vector(vectors[level-1]);
    //    mg_transfer.restrict_and_add(level,vectors[level-1],vectors[level]);
    //    pcout << vectors[level-1].l2_norm() << " ";
    //  }
    // pcout << std::endl;
    // preconditioner.vmult(search_direction, system_rhs);
    // pcout << search_direction.l2_norm() << std::endl;

    search_direction = 0;
    cg.solve(system_matrix,
             search_direction,
             system_rhs, /*PreconditionIdentity()*/
             preconditioner);
    time_solve += time.wall_time();
    linear_iterations += solver_control.last_step();

    pcout << "Time solve (" << solver_control.last_step() << " iterations)             "
          << time.wall_time() << " s\n";

    constraints.distribute(search_direction);

    time.restart();
    double       final_residual_norm = initial_residual_norm;
    double       alpha               = 1.;
    unsigned int n_steps             = 0;
    while (n_steps < 100)
      {
        tentative_solution = solution;
        tentative_solution.add(alpha, search_direction);
        system_matrix.compute_residual(system_rhs, tentative_solution);
        ++n_residual;
        final_residual_norm = system_rhs.l2_norm();
        if (final_residual_norm < initial_residual_norm)
          break;
        alpha = alpha / 2.;
        ++n_steps;
      }
    time_residual += time.wall_time();
    pcout << "Residual norm: " << initial_residual_norm << " in " << n_steps << " steps to "
          << final_residual_norm << std::endl;
    solution = tentative_solution;
  }



  template <int dim>
  void
  LaplaceProblem<dim>::output_results(const unsigned int cycle) const
  {
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    solution.update_ghost_values();
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    Vector<double> owner(triangulation.n_active_cells());
    owner = (double)Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector(owner, "owner");
    data_out.build_patches(mapping, 1, DataOut<dim>::curved_inner_cells);

    std::ostringstream filename;
    filename << "solution-" << cycle << "." << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
             << ".vtu";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          {
            std::ostringstream filename;
            filename << "solution-" << cycle << "." << i << ".vtu";

            filenames.push_back(filename.str().c_str());
          }
        std::string   master_name = "solution-" + Utilities::to_string(cycle) + ".pvtu";
        std::ofstream master_output(master_name.c_str());
        data_out.write_pvtu_record(master_output, filenames);
      }
  }



  template <int dim>
  void
  LaplaceProblem<dim>::run()
  {
    const unsigned int n_inner_iterations = 100;
    for (unsigned int cycle = 0; cycle < 9 - dim; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          {
            // GridGenerator::hyper_cube (triangulation, 0., 1.);
            GridGenerator::hyper_ball(triangulation);
            static const SphericalManifold<dim> boundary;
            triangulation.set_all_manifold_ids_on_boundary(0);
            triangulation.set_manifold(0, boundary);
            triangulation.refine_global(5 - dim);
            setup_system();
          }
        else
          {
            triangulation.set_all_refine_flags();
            triangulation.prepare_coarsening_and_refinement();
            // parallel::distributed::SolutionTransfer<dim,LinearAlgebra::distributed::Vector<number>
            // > solution_transfer(dof_handler);
            // solution_transfer.prepare_for_coarsening_and_refinement(solution);
            triangulation.execute_coarsening_and_refinement();
            setup_system();
            // solution_transfer.interpolate(solution);
          }

        interpolate_boundary_values();
        output_results(cycle * (1 + n_inner_iterations) + 0);
        n_residual                   = 0;
        linear_iterations            = 0;
        time_solve                   = 0;
        time_residual                = 0;
        unsigned int inner_iteration = 0;
        for (; inner_iteration < n_inner_iterations; ++inner_iteration)
          {
            solve(inner_iteration == 0);
            output_results(cycle * (1 + n_inner_iterations) + inner_iteration + 1);
            if (system_rhs.l2_norm() < 1e-12)
              break;
          }
        pcout << "Computing times: nl iterations: " << inner_iteration + 1 << " residuals "
              << n_residual << " " << time_residual << "  linear solver " << linear_iterations
              << " " << time_solve << std::endl;
        pcout << std::endl;
      };
  }
} // namespace multigrid



int
main(int argc, char *argv[])
{
  try
    {
      using namespace multigrid;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      LaplaceProblem<dimension> laplace_problem;
      // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      //  deallog.depth_console(3);
      laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------" << std::endl;
      return 1;
    }

  return 0;
}
