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
 * This program solves the Poisson equation on a shell with variable
 * coefficients, implementing a full multigrid cycle and a conjugate gradient
 * method preconditioned by a V-cycle. This program is similar to
 * ../poisson_cube/program.cc except for the geometry, variable coefficient,
 * and the solution. This code does not support adaptivity, see e.g. the
 * step-37 program of deal.II or ../poisson_l/program.cc for such a case.
 */


// First include the necessary files from the deal.II library.
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/tria.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "../common/multigrid_solver.h"


namespace multigrid
{
  using namespace dealii;

  // Here at the top of the file, we collect the main global settings. The
  // degree can be passed as the first argument to the program, but due to the
  // templates we need to precompile the respective programs. Here we specify
  // a minimum and maximum degree we want to support. Degrees outside this
  // range will not do any work.
  const unsigned int dimension      = 3;
  const unsigned int minimal_degree = 1;
  const unsigned int maximal_degree = 9;

  // We also select a mixed-precision approach as default. You can
  // independently change the number type for the outer iteration via
  // full_number and the number type for the multigrid v-cycle.
  using vcycle_number = float;
  using full_number   = double;

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution()
      : Function<dim>()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;

    virtual Tensor<1, dim>
    gradient(const Point<dim> &p, const unsigned int component = 0) const;


    virtual Tensor<1, dim, VectorizedArray<double>>
    gradient(const Point<dim, VectorizedArray<double>> &p, const unsigned int component = 0) const;

    virtual double
    laplacian(const Point<dim> &p, const unsigned int component = 0) const;

    virtual VectorizedArray<double>
    laplacian(const Point<dim, VectorizedArray<double>> &p, const unsigned int component = 0) const;
  };



  template <int dim>
  double
  Solution<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    return std::sin(2 * numbers::PI * (p[0] + p[1]));
  }

  template <int dim>
  Tensor<1, dim>
  Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
  {
    Tensor<1, dim> grad_tensor;
    grad_tensor[0] = 1;
    grad_tensor[1] = 1;
    return 2 * numbers::PI * std::cos(2 * numbers::PI * (p[0] + p[1])) * grad_tensor;
  }

  template <int dim>
  Tensor<1, dim, VectorizedArray<double>>
  Solution<dim>::gradient(const Point<dim, VectorizedArray<double>> &p, const unsigned int) const
  {
    Tensor<1, dim, VectorizedArray<double>> direction;
    for (int i = 0; i < 2; ++i)
      direction[i] = make_vectorized_array(1.0);

    return 2 * numbers::PI * std::cos(2 * numbers::PI * (p[0] + p[1])) * direction;
  }

  template <int dim>
  double
  Solution<dim>::laplacian(const Point<dim> &p, const unsigned int) const
  {
    return -2 * 2 * numbers::PI * 2 * numbers::PI * std::sin(2 * numbers::PI * (p[0] + p[1]));
  }

  template <int dim>
  VectorizedArray<double>
  Solution<dim>::laplacian(const Point<dim, VectorizedArray<double>> &p, const unsigned int) const
  {
    return -2 * 2 * numbers::PI * 2 * numbers::PI * std::sin(2 * numbers::PI * (p[0] + p[1]));
  }



  // function computing the coefficient
  template <int dim, typename Number = double>
  class Coefficient : public Function<dim, Number>
  {
  public:
    Coefficient()
      : Function<dim, Number>()
    {}

    virtual Number
    value(const Point<dim, Number> &p, const unsigned int component = 0) const;

    virtual Tensor<1, dim, Number>
    gradient(const Point<dim, Number> &p, const unsigned int component = 0) const;
  };

  template <int dim, typename Number>
  Number
  Coefficient<dim, Number>::value(const Point<dim, Number> &p, const unsigned int) const
  {
    Number prod = 1.0;
    for (int e = 0; e < dim; ++e)
      {
        Number c = std::cos(2 * numbers::PI * p[e] + 0.1 * e);
        prod *= c * c;
      }

    return 1.0 + 1.0e6 * prod;
  }



  template <int dim, typename Number>
  Tensor<1, dim, Number>
  Coefficient<dim, Number>::gradient(const Point<dim, Number> &p, const unsigned int) const
  {
    Tensor<1, dim, Number> prod;
    for (int d = 0; d < dim; ++d)
      {
        prod[d] = 1.0;

        for (int e = 0; e < dim; ++e)
          {
            Number c = std::cos(2 * numbers::PI * p[e] + 0.1 * e);

            if (e == d)
              {
                prod[d] *= -4 * numbers::PI * c * std::sin(2 * numbers::PI * p[e] + 0.1 * e);
              }
            else
              {
                prod[d] *= c * c;
              }
          }
      }

    return 1.0e6 * prod;
  }



  // function computing the right-hand side
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  private:
    Solution<dim>    solution;
    Coefficient<dim> coefficient;

  public:
    RightHandSide()
      : Function<dim>()
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const;
  };

  template <int dim>
  double
  RightHandSide<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    return -(solution.laplacian(p) * coefficient.value(p) +
             coefficient.gradient(p) * solution.gradient(p));
  }



  template <int dim, int degree_finite_element>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();
    void
    run(const unsigned int max_size,
        const unsigned int n_mg_cycles,
        const unsigned int n_pre_smooth,
        const unsigned int n_post_smooth);

  private:
    void
    setup_system();
    void
    solve(const unsigned int n_mg_cycles,
          const unsigned int n_pre_smooth,
          const unsigned int n_post_smooth);

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FE_Q<dim>            fe;
    DoFHandler<dim>      dof_handler;
    MappingQGeneric<dim> mapping;

    LinearAlgebra::distributed::Vector<double> solution;

    double             setup_time;
    ConditionalOStream pcout;

    ConvergenceTable convergence_table;
  };



  template <int dim, int degree_finite_element>
  LaplaceProblem<dim, degree_finite_element>::LaplaceProblem()
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
    fe(degree_finite_element)
    , dof_handler(triangulation)
    , mapping(std::min(10, degree_finite_element))
    , setup_time(0.)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}



  template <int dim, int degree_finite_element>
  void
  LaplaceProblem<dim, degree_finite_element>::setup_system()
  {
    Timer time;
    setup_time = 0;

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    setup_time += time.wall_time();

    pcout << "DoF setup time:        " << setup_time << "s" << std::endl;
  }



  template <int dim, int degree_finite_element>
  void
  LaplaceProblem<dim, degree_finite_element>::solve(const unsigned int n_mg_cycles,
                                                    const unsigned int n_pre_smooth,
                                                    const unsigned int n_post_smooth)
  {
    Solution<dim>                                                           analytic_solution;
    MultigridSolver<dim, degree_finite_element, vcycle_number, full_number> solver(
      dof_handler,
      analytic_solution,
      RightHandSide<dim>(),
      Coefficient<dim>(),
      n_pre_smooth,
      n_post_smooth,
      n_mg_cycles);

    Timer time;

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory =
      Utilities::MPI::min_max_avg(stats.VmRSS / 1024., MPI_COMM_WORLD);

    pcout << "Memory stats [MB]: " << memory.min << " [p" << memory.min_index << "] " << memory.avg
          << " " << memory.max << " [p" << memory.max_index << "]" << std::endl;

    double best_time = 1e10, tot_time = 0;
    for (unsigned int i = 0; i < 5; ++i)
      {
        time.reset();
        time.start();
        solver.solve(false);
        best_time = std::min(time.wall_time(), best_time);
        tot_time += time.wall_time();
        pcout << "Time solve   (CPU/wall)    " << time.cpu_time() << "s/" << time.wall_time()
              << "s\n";
      }
    const double              vcycl_reduction = solver.solve(true);
    Utilities::MPI::MinMaxAvg stat = Utilities::MPI::min_max_avg(tot_time, MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "All solver time " << stat.min << " [p" << stat.min_index << "] " << stat.avg
                << " " << stat.max << " [p" << stat.max_index << "]" << std::endl;
    solver.print_wall_times();

    const double l2_error = solver.compute_l2_error(triangulation.n_global_levels() - 1);

    time.restart();
    auto         cg_details  = solver.solve_cg();
    const double time_cg     = time.wall_time();
    const double l2_error_cg = solver.compute_l2_error(triangulation.n_global_levels() - 1);
    solver.print_wall_times();

    double best_mv = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          solver.do_matvec();
        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
        best_mv = std::min(best_mv, stat.max);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "matvec time dp " << stat.min << " [p" << stat.min_index << "] " << stat.avg
                    << " " << stat.max << " [p" << stat.max_index << "]"
                    << " DoFs/s: " << dof_handler.n_dofs() / stat.max << std::endl;
      }
    double best_mvs = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          solver.do_matvec_smoother();
        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
        best_mvs = std::min(best_mvs, stat.max);
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Best timings for ndof = " << dof_handler.n_dofs() << "   mv " << best_mv
                << "    mv smooth " << best_mvs << "   mg " << best_time << std::endl;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "L2 error with ndof = " << dof_handler.n_dofs() << "  " << l2_error
                << "  with CG " << l2_error_cg << std::endl;

    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("mv_outer", best_mv);
    convergence_table.add_value("mv_inner", best_mvs);
    convergence_table.add_value("reduction", vcycl_reduction);
    convergence_table.add_value("fmg_L2error", l2_error);
    convergence_table.add_value("fmg_time", best_time);
    convergence_table.add_value("cg_L2error", l2_error_cg);
    convergence_table.add_value("cg_time", time_cg);
    convergence_table.add_value("cg_its", cg_details.first);
    convergence_table.add_value("cg_reduction", cg_details.second);
  }



  template <int dim, int degree_finite_element>
  void
  LaplaceProblem<dim, degree_finite_element>::run(const unsigned int max_size,
                                                  const unsigned int n_mg_cycles,
                                                  const unsigned int n_pre_smooth,
                                                  const unsigned int n_post_smooth)
  {
    pcout << "Testing " << fe.get_name() << std::endl;

    for (unsigned int cycle = 0; cycle < 35; ++cycle)
      {
        triangulation.clear();
        pcout << "Cycle " << cycle << std::endl;

        triangulation.clear();
        // shell from a hexahedron
        if (cycle % 2 == 0)
          GridGenerator::hyper_shell(triangulation, Point<dim>(), 0.5, 1.0, 6);
        // shell from a rhmobic dodecahedron
        else
          GridGenerator::hyper_shell(triangulation, Point<dim>(), 0.5, 1.0, 12);

        triangulation.refine_global(((dim == 2) ? 2 : 0) + cycle / 2);

        setup_system();
        if (dof_handler.n_dofs() > max_size)
          {
            pcout << "Max size reached, terminating." << std::endl;
            pcout << std::endl;
            break;
          }

        solve(n_mg_cycles, n_pre_smooth, n_post_smooth);
        pcout << std::endl;
      };

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        convergence_table.set_scientific("fmg_L2error", true);
        convergence_table.set_precision("fmg_L2error", 3);
        convergence_table.evaluate_convergence_rates("fmg_L2error",
                                                     "cells",
                                                     ConvergenceTable::reduction_rate_log2,
                                                     dim);
        convergence_table.set_scientific("mv_outer", true);
        convergence_table.set_precision("mv_outer", 3);
        convergence_table.set_scientific("mv_inner", true);
        convergence_table.set_precision("mv_inner", 3);
        convergence_table.set_scientific("fmg_time", true);
        convergence_table.set_precision("fmg_time", 3);

        convergence_table.set_scientific("reduction", true);
        convergence_table.set_precision("reduction", 3);
        convergence_table.set_scientific("cg_L2error", true);
        convergence_table.set_precision("cg_L2error", 3);
        convergence_table.evaluate_convergence_rates("cg_L2error",
                                                     "cells",
                                                     ConvergenceTable::reduction_rate_log2,
                                                     dim);
        convergence_table.set_scientific("cg_reduction", true);
        convergence_table.set_precision("cg_reduction", 3);
        convergence_table.set_scientific("cg_time", true);
        convergence_table.set_precision("cg_time", 3);

        convergence_table.write_text(std::cout);

        std::cout << std::endl << std::endl;
      }
  }



  template <int dim, int min_degree, int max_degree>
  class LaplaceRunTime
  {
  public:
    LaplaceRunTime(const unsigned int target_degree,
                   const unsigned int max_size,
                   const unsigned int n_mg_cycles,
                   const unsigned int n_pre_smooth,
                   const unsigned int n_post_smooth)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim, min_degree> laplace_problem;
          laplace_problem.run(max_size, n_mg_cycles, n_pre_smooth, n_post_smooth);
        }
      LaplaceRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree> m(
        target_degree, max_size, n_mg_cycles, n_pre_smooth, n_post_smooth);
    }
  };
} // namespace multigrid



// @sect3{The <code>main</code> function}

// Apart from the fact that we set up the MPI framework according to step-40,
// there are no surprises in the main function.
int
main(int argc, char *argv[])
{
  try
    {
      using namespace multigrid;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);


      unsigned int degree        = numbers::invalid_unsigned_int;
      unsigned int maxsize       = numbers::invalid_unsigned_int;
      unsigned int n_mg_cycles   = 1;
      unsigned int n_pre_smooth  = 3;
      unsigned int n_post_smooth = 3;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout << "Expected at least one argument." << std::endl
                      << "Usage:" << std::endl
                      << "./program degree maxsize n_mg_cycles n_pre_smooth n_post_smooth"
                      << std::endl
                      << "The parameters degree to n_post_smooth are integers, "
                      << "the last selects between a square mesh or a doubling mesh" << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        maxsize = std::atoi(argv[2]);
      if (argc > 3)
        n_mg_cycles = std::atoi(argv[3]);
      if (argc > 4)
        n_pre_smooth = std::atoi(argv[4]);
      if (argc > 5)
        n_post_smooth = std::atoi(argv[5]);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Number of MG cycles in V-cycle: " << n_mg_cycles << std::endl
                  << "Number of pre-smoother iters:   " << n_pre_smooth << std::endl
                  << "Number of post-smoother iters:  " << n_post_smooth << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(
        degree, maxsize, n_mg_cycles, n_pre_smooth, n_post_smooth);
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
