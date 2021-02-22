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
 * This program solves the Poisson equation on an L-shaped domain with
 * adaptive meshes, using a conjugate gradient method preconditioned by a
 * V-cycle.
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include "../common/laplace_operator.h"

#include <iostream>
#include <fstream>
#include <sstream>


namespace multigrid
{
  using namespace dealii;

  // Here at the top of the file, we collect the main global settings. The
  // degree can be passed as the first argument to the program, but due to the
  // templates we need to precompile the respective programs. Here we specify
  // a minimum and maximum degree we want to support. Degrees outside this
  // range will not do any work.
  const unsigned int dimension = 3;
  const unsigned int minimal_degree = 1;
  const unsigned int maximal_degree = 9;

  // We also select a mixed-precision approach as default. You can
  // independently change the number type for the outer iteration via
  // full_number and the number type for the multigrid v-cycle.
  using vcycle_number = float;
  using full_number = double;

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution(const bool use_hyper_l) : Function<dim> (1), hyper_l(use_hyper_l) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const
    {
      if (hyper_l == false)
        {
          Point<2> p2;
          for (unsigned int d=0; d<2; ++d)
            p2[d] = p[d];
          Functions::LSingularityFunction func;
          return func.value(p2, component);
        }
      else
        return 0;
    }

    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const
    {
      if (hyper_l == false)
        {
          Point<2> p2;
          for (unsigned int d=0; d<2; ++d)
            p2[d] = p[d];
          Functions::LSingularityFunction func;
          Tensor<1,2> grad = func.gradient(p2, component);
          Tensor<1,dim> outgrad;
          for (unsigned int d=0; d<2; ++d)
            outgrad[d] = grad[d];
          return outgrad;
        }
      else
        return Tensor<1,dim>();
    }

  private:
    const bool hyper_l;
  };

  template <>
  class Solution<2> : public Functions::LSingularityFunction
  {
  public:
    Solution(const bool) : Functions::LSingularityFunction() {}
  };



  template <int dim,int degree_finite_element>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    void run (const unsigned int max_size,
              const unsigned int n_smoother,
              const bool use_hyper_l);

  private:
    void setup_system ();
    std::pair<unsigned int,std::pair<double,double> > solve (const unsigned int n_smoother,
                                                             const bool first_time);
    void output_results (const unsigned int cycle) const;
    void interpolate_boundary_values(const bool use_hyper_l);

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim>  triangulation;
#else
    Triangulation<dim>                         triangulation;
#endif

    MappingQGeneric<dim>                       mapping;
    FE_Q<dim>                                  fe;
    DoFHandler<dim>                            dof_handler;

    AffineConstraints<double>                  constraints_without_dirichlet;
    AffineConstraints<double>                  constraints;
    std::shared_ptr<MatrixFree<dim,full_number> > system_matrix_free;
    typedef LaplaceOperator<dim,degree_finite_element,full_number> SystemMatrixType;
    SystemMatrixType                           system_matrix;

    MGConstrainedDoFs                          mg_constrained_dofs;
    MGTransferMatrixFree<dim,vcycle_number>     mg_transfer;
    typedef LaplaceOperator<dim,degree_finite_element,vcycle_number>  LevelMatrixType;
    MGLevelObject<LevelMatrixType>             mg_matrices;

    LinearAlgebra::distributed::Vector<full_number> solution;
    LinearAlgebra::distributed::Vector<full_number> solution_update;
    LinearAlgebra::distributed::Vector<full_number> system_rhs;

    double                                     setup_time;
    ConditionalOStream                         pcout;
  };



  template <int dim, int degree>
  LaplaceProblem<dim, degree>::LaplaceProblem ()
    :
#ifdef DEAL_II_WITH_P4EST
    triangulation(MPI_COMM_WORLD,
                  Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
#else
    triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
#endif
    mapping (degree),
    fe (degree),
    dof_handler (triangulation),
    pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}



  template <int dim, int degree>
  void LaplaceProblem<dim, degree>::setup_system ()
  {
    Timer time;
    time.start ();
    setup_time = 0;

    system_matrix_free.reset(new MatrixFree<dim,full_number>());
    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs (fe);
    dof_handler.distribute_mg_dofs ();

    pcout << "Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << std::endl;

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim>(),
                                              constraints);
    constraints.close();
    constraints_without_dirichlet.clear();
    constraints_without_dirichlet.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints_without_dirichlet);
    constraints_without_dirichlet.close();
    setup_time += time.wall_time();
    pcout << "Distribute DoFs & B.C.                "
          << time.wall_time() << "s" << std::endl;
    time.restart();

    std::vector<const DoFHandler<dim> *> dof_handlers(2, &dof_handler);
    {
      std::vector<const AffineConstraints<double> *> constraint(2);
      constraint[0] = &constraints;
      constraint[1] = &constraints_without_dirichlet;
      typename MatrixFree<dim,full_number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim,full_number>::AdditionalData::none;
      additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                              update_quadrature_points);
      system_matrix_free->reinit (mapping, dof_handlers, constraint,
                                  QGauss<1>(fe.degree+1), additional_data);
      std::vector<unsigned int> mask(1);
      system_matrix.initialize (system_matrix_free, mask);
      system_matrix.evaluate_coefficient(Functions::ConstantFunction<dim>(1.));
    }
    system_matrix_free->initialize_dof_vector(solution, 1);
    system_matrix.initialize_dof_vector(solution_update);
    system_matrix.initialize_dof_vector(system_rhs);

    setup_time += time.wall_time();
    pcout << "Setup matrix-free system              "
          << time.wall_time() << "s" << std::endl;
    time.restart();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels-1);

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

    for (unsigned int level=0; level<nlevels; ++level)
      {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level,
                                                      relevant_dofs);
        AffineConstraints<double> level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();
        std::vector<const AffineConstraints<double> *> constraint(2);
        AffineConstraints<double> dummy;
        dummy.close();
        constraint[0] = &level_constraints;
        constraint[1] = &dummy;

        typename MatrixFree<dim,vcycle_number>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
          MatrixFree<dim,vcycle_number>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                update_quadrature_points);
        additional_data.mg_level = level;
        std::shared_ptr<MatrixFree<dim,vcycle_number> > mg_matrix_free_level(new MatrixFree<dim,vcycle_number>());
        mg_matrix_free_level->reinit(mapping, dof_handlers, constraint,
                                     QGauss<1>(fe.degree+1), additional_data);

        mg_matrices[level].initialize(mg_matrix_free_level, mg_constrained_dofs,
                                      level);
        mg_matrices[level].evaluate_coefficient(Functions::ConstantFunction<dim>(1.));
      }
    setup_time += time.wall_time();
    pcout << "Setup matrix-free levels              "
          << time.wall_time() << "s" << std::endl;

    time.restart();
    mg_transfer.clear();
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(dof_handler);
    setup_time += time.wall_time();
    pcout << "MG build transfer time                "
          << time.wall_time() << "s" << std::endl;
    pcout << "Total setup time               (wall) " << setup_time
          << "s\n";
  }



  template <int dim,int degree>
  void LaplaceProblem<dim,degree>::interpolate_boundary_values(const bool use_hyper_l)
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(mapping, dof_handler, 0, Solution<dim>(use_hyper_l), boundary_values);
    for (typename std::map<types::global_dof_index, double>::iterator it = boundary_values.begin();
         it != boundary_values.end(); ++it)
      if (solution.locally_owned_elements().is_element(it->first))
        solution(it->first) = it->second;
    constraints_without_dirichlet.distribute(solution);
  }




  template <int dim,int degree>
  std::pair<unsigned int,std::pair<double,double> >
  LaplaceProblem<dim,degree>::solve (const unsigned int n_smoother,
                                     const bool /*first_time*/)
  {
    Timer time;
    typedef PreconditionChebyshev<LevelMatrixType,LinearAlgebra::distributed::Vector<vcycle_number> > SmootherType;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<vcycle_number> >
    mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
      {
        if (level > 0)
          {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = n_smoother;
            smoother_data[level].eig_cg_n_iterations = 15;
          }
        else
          {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
          }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
      }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<vcycle_number> > mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<vcycle_number> > mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType> > mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<vcycle_number> > mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<vcycle_number> > mg(mg_matrix,
                                                                     mg_coarse,
                                                                     mg_transfer,
                                                                     mg_smoother,
                                                                     mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LinearAlgebra::distributed::Vector<vcycle_number>,
                   MGTransferMatrixFree<dim,vcycle_number> >
                   preconditioner(dof_handler, mg, mg_transfer);

    ReductionControl solver_control (system_matrix.m(), 1e-13, 1e-9);
    SolverCG<LinearAlgebra::distributed::Vector<full_number> > cg (solver_control);
    pcout << "MG build smoother time                "
          << time.wall_time() << "s" << std::endl;

    time.reset();
    time.start();
    solution_update = 0;
    cg.solve (system_matrix, solution_update, system_rhs,
              preconditioner);

    std::pair<unsigned int,std::pair<double,double> > stats(solver_control.last_step(),
                                                            std::pair<double,double>(time.wall_time(),0));
    time.reset();
    time.start();
    solution_update = 0;
    cg.solve (system_matrix, solution_update, system_rhs,
              preconditioner);
    stats.second.second = time.wall_time();
    pcout << "Time solve ("
          << stats.first
          << " iterations)             "
          << stats.second.second << "s  convergence rate "
          << std::pow(solver_control.last_value()/solver_control.initial_value(), 1./stats.first) << std::endl;

    constraints.distribute(solution_update);
    solution += solution_update;

    return stats;
  }




  template <int dim, int degree>
  void LaplaceProblem<dim,degree>::output_results (const unsigned int cycle) const
  {
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    solution.update_ghost_values();
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    Vector<double> owner(triangulation.n_active_cells());
    owner = (double)Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    data_out.add_data_vector (owner, "owner");
    data_out.build_patches (mapping, 1, DataOut<dim>::curved_inner_cells);

    std::ostringstream filename;
    filename << "solution-"
             << cycle
             << "." << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
             << ".vtu";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
          {
            std::ostringstream filename;
            filename << "solution-"
                     << cycle
                     << "."
                     << i
                     << ".vtu";

            filenames.push_back(filename.str().c_str());
          }
        std::string master_name = "solution-" + Utilities::to_string(cycle) + ".pvtu";
        std::ofstream master_output (master_name.c_str());
        data_out.write_pvtu_record (master_output, filenames);
      }
  }



  void create_tria (Triangulation<2> &tria,
                    const bool)
  {
    GridGenerator::hyper_L (tria);
  }



  void create_tria (Triangulation<3> &tria,
                    const bool use_hyper_l)
  {
    if (use_hyper_l)
      {
        GridGenerator::hyper_L (tria);
        for (Triangulation<3>::cell_iterator cell=tria.begin();
             cell != tria.end(); ++cell)
          for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary() &&
                std::abs(cell->face(f)->center()[0]+1) < 1e-10)
              cell->face(f)->set_boundary_id(0);
            else if (cell->face(f)->at_boundary())
              cell->face(f)->set_boundary_id(1);
      }
    else
      {
        Triangulation<2> tria_2d;
        GridGenerator::hyper_L (tria_2d);
        GridGenerator::extrude_triangulation(tria_2d, 2, 1., tria);
        for (Triangulation<3>::cell_iterator cell=tria.begin();
             cell != tria.end(); ++cell)
          for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary())
              cell->face(f)->set_boundary_id(0);
      }
  }



  template <int dim, int degree>
  void LaplaceProblem<dim,degree>::run (const unsigned int max_size,
                                        const unsigned int n_smoother,
                                        const bool use_hyper_l)
  {
    ConvergenceTable     convergence_table;
    for (unsigned int cycle=0; cycle<20; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          {
            create_tria(triangulation, use_hyper_l);
            if (dim == 2)
              triangulation.refine_global (5);
            else
              triangulation.refine_global (3);

            setup_system();
          }
        else
          {
            Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
            IndexSet relevant_dofs;
            DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
            LinearAlgebra::distributed::Vector<double> ghosted_vec
              (solution.locally_owned_elements(), relevant_dofs, MPI_COMM_WORLD);
            ghosted_vec = solution;
            ghosted_vec.update_ghost_values();
            KellyErrorEstimator<dim>::estimate (dof_handler,
                                                QGauss<dim-1>(fe.degree+1),
                                                std::map<types::boundary_id, const Function<dim> *>(),
                                                ghosted_vec,
                                                estimated_error_per_cell);
            parallel::distributed::GridRefinement::
              refine_and_coarsen_fixed_number (triangulation,
                                               estimated_error_per_cell,
                                               (use_hyper_l ? 0.15 : 0.3), 0.03);
            triangulation.prepare_coarsening_and_refinement ();
            parallel::distributed::SolutionTransfer<dim,LinearAlgebra::distributed::Vector<full_number> > solution_transfer(dof_handler);
            solution_transfer.prepare_for_coarsening_and_refinement(solution);
            triangulation.execute_coarsening_and_refinement();
            setup_system ();
            //solution_transfer.interpolate(solution);
          }

        interpolate_boundary_values(use_hyper_l);
        system_matrix.compute_residual(system_rhs, solution,
                                       use_hyper_l ? Functions::ConstantFunction<dim>(1.) :
                                       Functions::ZeroFunction<dim>());

        std::pair<unsigned int, std::pair<double,double> > stats = solve(n_smoother, false);

        output_results(cycle);

        solution.update_ghost_values();
        Vector<float> error_per_cell(triangulation.n_active_cells());
        VectorTools::integrate_difference (mapping,
                                           dof_handler,
                                           solution,
                                           Solution<dim>(use_hyper_l),
                                           error_per_cell,
                                           QGauss<dim>(fe.degree+2),
                                           VectorTools::L2_norm);
        const double L2_error =
          std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), MPI_COMM_WORLD));
        VectorTools::integrate_difference (mapping,
                                           dof_handler,
                                           solution,
                                           Solution<dim>(use_hyper_l),
                                           error_per_cell,
                                           QGauss<dim>(fe.degree+1),
                                           VectorTools::H1_seminorm);
        const double grad_error =
          std::sqrt(Utilities::MPI::sum(error_per_cell.norm_sqr(), MPI_COMM_WORLD));
        convergence_table.add_value("cells", triangulation.n_global_active_cells());
        convergence_table.add_value("dofs",  dof_handler.n_dofs());
        convergence_table.add_value("val_L2",    L2_error);
        convergence_table.add_value("grad_L2",   grad_error);
        convergence_table.add_value("solver_its", stats.first);
        convergence_table.add_value("solver_time1", stats.second.first);
        convergence_table.add_value("solver_time2", stats.second.second);

        pcout << std::endl;

        if (dof_handler.n_dofs() > max_size)
          {
            pcout << "Max size reached, terminating." << std::endl;
            pcout << std::endl;
            break;
          }
      };
    convergence_table.set_precision("val_L2", 3);
    convergence_table.set_scientific("val_L2", true);
    convergence_table.set_precision("grad_L2", 3);
    convergence_table.set_scientific("grad_L2", true);
    convergence_table.set_precision("solver_time1", 3);
    convergence_table.set_scientific("solver_time1", true);
    convergence_table.set_precision("solver_time2", 3);
    convergence_table.set_scientific("solver_time2", true);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      convergence_table.write_text(std::cout);
  }



  template <int dim, int min_degree, int max_degree>
  class LaplaceRunTime {
  public:
    LaplaceRunTime(const unsigned int target_degree,
                   const unsigned int max_size,
                   const unsigned int n_smoother,
                   const bool use_hyper_l)
    {
      if (min_degree>max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim,min_degree> laplace_problem;
          laplace_problem.run(max_size, n_smoother, use_hyper_l);
        }
      LaplaceRunTime<dim,(min_degree<=max_degree?(min_degree+1):min_degree),max_degree>
                     m(target_degree, max_size, n_smoother, use_hyper_l);
    }
  };
}




int main (int argc, char *argv[])
{
  try
    {
      using namespace multigrid;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      unsigned int degree = numbers::invalid_unsigned_int;
      unsigned int maxsize = numbers::invalid_unsigned_int;
      unsigned int n_smoother = 3;
      bool use_hyper_l = false;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Expected at least one argument." << std::endl
                      << "Usage:" << std::endl
                      << "./program degree maxsize n_smoother hyper_l"
                      << std::endl
                      << "The parameters degree to n_smoother are integers, "
                      << "the last selects between a hyper_l (h) or a 2D L (other string)"
                      << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        maxsize = std::atoi(argv[2]);
      if (argc > 3)
        n_smoother = std::atoi(argv[3]);
      if (argc > 4)
        use_hyper_l = argv[4][0] == 'h';

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Number of smoother iterations:  " << n_smoother << std::endl
                  << "Use hyper-l test case:          " << use_hyper_l << std::endl
                  << std::endl;

      LaplaceRunTime<dimension,minimal_degree,maximal_degree> run(degree, maxsize,
                                                                  n_smoother,
                                                                  use_hyper_l);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
