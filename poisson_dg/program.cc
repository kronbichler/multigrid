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
 * This program solves the Poisson equation on a cube with the full multigrid
 * cycle and a conjugate gradient method preconditioned by a V-cycle. This
 * code does not support adaptivity, see e.g. the step-37 program of deal.II
 * or ../poisson_l/program.cc for such a case.
 */


// First include the necessary files from the deal.II library.
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_dgq.h>
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


#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif


#include "../common/multigrid_solver_dg.h"


namespace multigrid
{
  using namespace dealii;

  // Here at the top of the file, we collect the main global settings. The
  // degree can be passed as the first argument to the program, but due to the
  // templates we need to precompile the respective programs. Here we specify
  // a minimum and maximum degree we want to support. Degrees outside this
  // range will not do any work.
  const unsigned int dimension      = 3;
  const unsigned int minimal_degree = 3;
  const unsigned int maximal_degree = 6;
  const double       wave_number    = 3.;
  const bool         deform_grid    = false;

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
  };



  template <int dim>
  double
  Solution<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    double val = 1;
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::sin(numbers::PI * p[d] * wave_number);
    return val;
  }



  template <int dim>
  Tensor<1, dim>
  Solution<dim>::gradient(const Point<dim> &p, const unsigned int) const
  {
    Tensor<1, dim> return_value;
    for (unsigned int d = 0; d < dim; ++d)
      {
        return_value[d] = numbers::PI * wave_number * std::cos(numbers::PI * p[d] * wave_number);
        for (unsigned int e = 0; e < dim; ++e)
          if (d != e)
            return_value[d] *= std::sin(numbers::PI * p[d] * wave_number);
      }
    return return_value;
  }



  template <int dim>
  class RightHandSide : public Function<dim>
  {
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
    Solution<dim> sol;
    return dim * numbers::PI * wave_number * numbers::PI * wave_number * sol.value(p);
  }



  template <int dim, int degree_finite_element>
  class LaplaceProblem
  {
  public:
    LaplaceProblem();
    void
    run(const std::size_t  min_size,
        const std::size_t  max_size,
        const unsigned int n_mg_cycles,
        const unsigned int n_pre_smooth,
        const unsigned int n_post_smooth,
        const bool         use_doubling_mesh,
        const double       tolerance);

  private:
    void
    setup_system();
    void
    solve(const unsigned int n_mg_cycles,
          const unsigned int n_pre_smooth,
          const unsigned int n_post_smooth,
          const double       tolerance);

#ifdef DEAL_II_WITH_P4EST
    parallel::distributed::Triangulation<dim> triangulation;
#else
    Triangulation<dim> triangulation;
#endif

    FE_DGQHermite<dim>   fe;
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

    setup_time += time.wall_time();

    print_time(time.wall_time(), "Time distribute DG dofs", MPI_COMM_WORLD);
  }



  template <int dim, int degree_finite_element>
  void
  LaplaceProblem<dim, degree_finite_element>::solve(const unsigned int n_mg_cycles,
                                                    const unsigned int n_pre_smooth,
                                                    const unsigned int n_post_smooth,
                                                    const double       tolerance)
  {
    Solution<dim>                                                             analytic_solution;
    MultigridSolverDG<dim, degree_finite_element, vcycle_number, full_number> solver(
      dof_handler,
      analytic_solution,
      RightHandSide<dim>(),
      Functions::ConstantFunction<dim>(1.),
      n_pre_smooth,
      n_post_smooth,
      1);

    Timer time;

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    print_time(stats.VmRSS / 1024., "Memory stats [MB]", MPI_COMM_WORLD);

#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START("cg_solver");
#endif
    double                    time_cg = 1e10;
    std::pair<double, double> cg_details;
    for (unsigned int i = 0; i < std::max(4U, n_mg_cycles); ++i)
      {
        time.restart();
        cg_details = solver.solve_cg(tolerance);
        time_cg    = std::min(time.wall_time(), time_cg);
        print_time(time.wall_time(), "Time solve CG", MPI_COMM_WORLD);
      }
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP("cg_solver");
#endif
    const double l2_error_cg = solver.compute_l2_error();
    solver.print_wall_times();

    solver.print_matvec_details();
    double best_mv = 1e10;
    for (unsigned int i = 0; i < 5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 200 : 50;
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_START("matvec");
#endif
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          solver.do_matvec();
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP("matvec");
#endif
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
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_START("matvec_sp");
#endif
        time.restart();
        for (unsigned int i = 0; i < n_mv; ++i)
          solver.do_matvec_smoother();
#ifdef LIKWID_PERFMON
        LIKWID_MARKER_STOP("matvec_sp");
#endif
        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg(time.wall_time() / n_mv, MPI_COMM_WORLD);
        best_mvs = std::min(best_mvs, stat.max);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "matvec time sp " << stat.min << " [p" << stat.min_index << "] " << stat.avg
                    << " " << stat.max << " [p" << stat.max_index << "]"
                    << " DoFs/s: " << dof_handler.n_dofs() / stat.max << std::endl;
      }
    solver.print_matvec_details();
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Best timings for ndof = " << dof_handler.n_dofs() << "   mv " << best_mv
                << "    mv smooth " << best_mvs << "   cg-mg " << time_cg << std::endl;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "L2 error with ndof = " << dof_handler.n_dofs() << "  " << l2_error_cg
                << std::endl;

    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("mv_outer", best_mv);
    convergence_table.add_value("mv_inner", best_mvs);
    convergence_table.add_value("cg_L2error", l2_error_cg);
    convergence_table.add_value("cg_time", time_cg);
    convergence_table.add_value("cg_its", cg_details.first);
    convergence_table.add_value("cg_reduction", cg_details.second);
  }



  template <int dim, int degree_finite_element>
  void
  LaplaceProblem<dim, degree_finite_element>::run(const std::size_t  min_size,
                                                  const std::size_t  max_size,
                                                  const unsigned int n_mg_cycles,
                                                  const unsigned int n_pre_smooth,
                                                  const unsigned int n_post_smooth,
                                                  const bool         use_doubling_mesh,
                                                  const double       tolerance)
  {
    pcout << "Testing " << fe.get_name() << std::endl;
    const unsigned int sizes[] = {1,   2,   3,   4,   5,   6,   7,   8,   10,  12,   14,   16,  20,
                                  24,  28,  32,  40,  48,  56,  64,  80,  96,  112,  128,  160, 192,
                                  224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536};

    for (unsigned int cycle = 0; cycle < sizeof(sizes) / sizeof(unsigned int); ++cycle)
      {
        triangulation.clear();
        pcout << "Cycle " << cycle << std::endl;

        Timer                        time;
        std::size_t                  projected_size = numbers::invalid_size_type;
        unsigned int                 n_refine       = 0;
        Tensor<1, dim, unsigned int> mesh_sizes;
        if (use_doubling_mesh)
          {
            n_refine                     = cycle / dim;
            const unsigned int remainder = cycle % dim;
            Point<dim>         p1;
            for (unsigned int d = 0; d < dim; ++d)
              p1[d] = -1;
            Point<dim> p2;
            for (unsigned int d = 0; d < remainder; ++d)
              p2[d] = 3;
            for (unsigned int d = remainder; d < dim; ++d)
              p2[d] = 1;
            std::vector<unsigned int> subdivisions(dim, 1);
            for (unsigned int d = 0; d < remainder; ++d)
              subdivisions[d] = 2;
            const unsigned int base_refine = (1 << n_refine);
            projected_size                 = 1;
            for (unsigned int d = 0; d < dim; ++d)
              {
                mesh_sizes[d] = base_refine * subdivisions[d];
                projected_size *= base_refine * subdivisions[d] * (degree_finite_element + 1);
              }
            GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);
          }
        else
          {
            n_refine              = 0;
            unsigned int n_subdiv = sizes[cycle];
            if (n_subdiv > 1)
              while (n_subdiv % 2 == 0)
                {
                  n_refine += 1;
                  n_subdiv /= 2;
                }
            if (dim == 2)
              n_refine += 3;
            GridGenerator::subdivided_hyper_cube(triangulation, n_subdiv, -0.9, 1.0);
            const unsigned int base_refine = (1 << n_refine);
            for (unsigned int d = 0; d < dim; ++d)
              mesh_sizes[d] = base_refine * n_subdiv;
            projected_size =
              Utilities::pow(base_refine * n_subdiv * (degree_finite_element + 1), dim);
          }

        if (projected_size < min_size)
          continue;

        if (projected_size > max_size)
          {
            pcout << "Projected size " << projected_size << " higher than max size, terminating."
                  << std::endl;
            pcout << std::endl;
            break;
          }

        triangulation.refine_global(n_refine);
        print_time(time.wall_time(), "Time create grid", MPI_COMM_WORLD);

        setup_system();

        std::locale s = pcout.get_stream().getloc();
        pcout.get_stream().imbue(std::locale("en_US.UTF-8"));
        pcout << "Number of degrees of freedom  " << dof_handler.n_dofs() << " = (";
        if (use_doubling_mesh)
          for (unsigned int d = 0; d < dim; ++d)
            pcout << mesh_sizes[d] << (d < dim - 1 ? " x " : ")");
        else
          pcout << mesh_sizes[0] << ")^" << dim;
        pcout << " x " << fe.degree + 1 << "^" << dim << std::endl;
        pcout.get_stream().imbue(s);

        solve(n_mg_cycles, n_pre_smooth, n_post_smooth, tolerance);
        pcout << std::endl;
      }

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        convergence_table.set_scientific("mv_outer", true);
        convergence_table.set_precision("mv_outer", 3);
        convergence_table.set_scientific("mv_inner", true);
        convergence_table.set_precision("mv_inner", 3);

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
                   const std::size_t  min_size,
                   const std::size_t  max_size,
                   const unsigned int n_mg_cycles,
                   const unsigned int n_pre_smooth,
                   const unsigned int n_post_smooth,
                   const bool         use_doubling_mesh,
                   const double       tolerance)
    {
      if (min_degree > max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim, min_degree> laplace_problem;
          laplace_problem.run(min_size,
                              max_size,
                              n_mg_cycles,
                              n_pre_smooth,
                              n_post_smooth,
                              use_doubling_mesh,
                              tolerance);
        }
      LaplaceRunTime<dim, (min_degree <= max_degree ? (min_degree + 1) : min_degree), max_degree> m(
        target_degree,
        min_size,
        max_size,
        n_mg_cycles,
        n_pre_smooth,
        n_post_smooth,
        use_doubling_mesh,
        tolerance);
    }
  };
} // namespace multigrid



// @sect3{The <code>main</code> function}

// Apart from the fact that we set up the MPI framework according to step-40,
// there are no surprises in the main function.
int
main(int argc, char *argv[])
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
  LIKWID_MARKER_THREADINIT;
#endif

  try
    {
      using namespace multigrid;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      unsigned int degree            = numbers::invalid_unsigned_int;
      std::size_t  maxsize           = static_cast<std::size_t>(-1);
      std::size_t  minsize           = 1;
      unsigned int n_mg_cycles       = 1;
      unsigned int n_pre_smooth      = 3;
      unsigned int n_post_smooth     = 3;
      bool         use_doubling_mesh = true;
      double       tolerance         = 1e-3;
      if (argc == 1)
        {
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
            std::cout
              << "Expected at least one argument." << std::endl
              << "Usage:" << std::endl
              << "./program degree minsize maxsize n_mg_cycles n_pre_smooth n_post_smooth doubling tolerance"
              << std::endl
              << "The parameters degree to n_post_smooth are integers, "
              << "the last selects between a square mesh or a doubling mesh" << std::endl;
          return 1;
        }

      if (argc > 1)
        degree = std::atoi(argv[1]);
      if (argc > 2)
        minsize = std::atoll(argv[2]);
      if (argc > 3)
        maxsize = std::atoll(argv[3]);
      if (argc > 4)
        n_mg_cycles = std::atoi(argv[4]);
      if (argc > 5)
        n_pre_smooth = std::atoi(argv[5]);
      if (argc > 6)
        n_post_smooth = std::atoi(argv[6]);
      if (argc > 7)
        use_doubling_mesh = argv[7][0] == 'd';
      if (argc > 8)
        tolerance = std::atof(argv[8]);

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << "Settings of parameters: " << std::endl
                  << "Number of MPI ranks:            "
                  << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl
                  << "Polynomial degree:              " << degree << std::endl
                  << "Minimum size:                   " << minsize << std::endl
                  << "Maximum size:                   " << maxsize << std::endl
                  << "Number of CG solutions:         " << n_mg_cycles << std::endl
                  << "Number of pre-smoother iters:   " << n_pre_smooth << std::endl
                  << "Number of post-smoother iters:  " << n_post_smooth << std::endl
                  << "Use doubling mesh:              " << use_doubling_mesh << std::endl
                  << "CG solver tolerance:            " << tolerance << std::endl
                  << std::endl;

      LaplaceRunTime<dimension, minimal_degree, maximal_degree> run(degree,
                                                                    minsize,
                                                                    maxsize,
                                                                    n_mg_cycles,
                                                                    n_pre_smooth,
                                                                    n_post_smooth,
                                                                    use_doubling_mesh,
                                                                    tolerance);
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

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
