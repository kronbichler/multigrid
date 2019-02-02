/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2016-2019 by Karl Ljungkvist and Martin Kronbichler
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
 * library, www.dealii.org (LGPL license), and ported to a GPU code.
 *
 * This code is designed to be integrated with the GPU code at
 * https://github.com/kalj/dealii-cuda
 *
 * The functionality of that code is, as of February 2019, already partly
 * integrated into the deal.II library and will be completed during the coming
 * months. At that point, this program will be ported to the deal.II
 * functionality.
 *
 * This program solves the Poisson equation on a cube with the full multigrid
 * cycle and a conjugate gradient method preconditioned by a V-cycle.
 */


// First include the necessary files from the deal.II library.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

// This includes the data structures for the efficient implementation of
// matrix-free methods or more generic finite element operators with the class
// MatrixFree.
#include <deal.II/matrix_free/operators.h>

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/gpu_array.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/mg_transfer_matrix_free_gpu.h"
#include "matrix_free_gpu/constraint_handler_gpu.h"
#include "matrix_free_gpu/matrix_free_gpu.h"
#include "matrix_free_gpu/fee_gpu.cuh"
#include "laplace_operator_gpu.h"

#include <iostream>
#include <fstream>
#include <sstream>


//#define CURVED_GRID 1

namespace Step37
{
  using namespace dealii;


  // To be efficient, the operations performed in the matrix-free
  // implementation require knowledge of loop lengths at compile time, which
  // are given by the degree of the finite element. Hence, we collect the
  // values of the two template parameters that can be changed at one place in
  // the code. Of course, one could make the degree of the finite element a
  // run-time parameter by compiling the computational kernels for all degrees
  // that are likely (say, between 1 and 6) and selecting the appropriate
  // kernel at run time. Here, we simply choose second order $Q_2$ elements
  // and choose dimension 3 as standard.
  const unsigned int dimension = 3;
  const unsigned int minimal_degree = 1;
  const unsigned int maximal_degree = 7;
  const double wave_number = 3.;
  using level_number = float;
  using full_number = double;

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;
  };



  template <int dim>
  double Solution<dim>::value (const Point<dim>   &p,
                               const unsigned int) const
  {
    double val = 1;
    for (unsigned int d=0; d<dim; ++d)
      val *= std::sin(numbers::PI*p[d]*wave_number);
    return val;
  }



  template <int dim>
  Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                         const unsigned int) const
  {
    Tensor<1,dim> return_value;
    for (unsigned int d=0; d<dim; ++d)
      {
        return_value[d] = numbers::PI*wave_number*std::cos(numbers::PI*p[d]*wave_number);
        for (unsigned int e=0; e<dim; ++e)
          if (d!=e)
            return_value[d] *= std::sin(numbers::PI*p[d]*wave_number);
      }
    return return_value;
  }



  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };


  template <int dim>
  double RightHandSide<dim>::value (const Point<dim>   &p,
                                    const unsigned int) const
  {
    Solution<dim> sol;
    return dim*numbers::PI*wave_number*numbers::PI*wave_number*sol.value(p);
  }




  // @sect3{LaplaceProblem class}

  // This class is based on the one in step-16. However, we replaced the
  // SparseMatrix<double> class by our matrix-free implementation, which means
  // that we can also skip the sparsity patterns. Notice that we define the
  // LaplaceOperator class with the degree of finite element as template
  // argument (the value is defined at the top of the file), and that we use
  // float numbers for the multigrid level matrices.
  //
  // The class also has a member variable to keep track of all the detailed
  // timings for setting up the entire chain of data before we actually go
  // about solving the problem. In addition, there is an output stream (that
  // is disabled by default) that can be used to output details for the
  // individual setup operations instead of the summary only that is printed
  // out by default.
  //
  // Since this program is designed to be used with MPI, we also provide the
  // usual @p pcout output stream that only prints the information of the
  // processor with MPI rank 0. The grid used for this programs can either be
  // a distributed triangulation based on p4est (in case deal.II is configured
  // to use p4est), otherwise it is a serial grid that only runs without MPI.
  template <int dim,int degree_finite_element>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    void run (const unsigned int max_size,
              const unsigned int n_mg_cycles,
              const unsigned int n_pre_smooth,
              const unsigned int n_post_smooth);

  private:
    void setup_system ();
    void solve (const unsigned int n_mg_cycles,
                const unsigned int n_pre_smooth,
                const unsigned int n_post_smooth);
    void output_results (const unsigned int cycle) const;

    Triangulation<dim>                         triangulation;

    FE_Q<dim>                                  fe;
    DoFHandler<dim>                            dof_handler;
    MappingQGeneric<dim>                       mapping;

    Vector<double>                             solution;

    double                                     setup_time;
    ConditionalOStream                         pcout;
    ConditionalOStream                         time_details;

    ConvergenceTable                           convergence_table;
  };



  // When we initialize the finite element, we of course have to use the
  // degree specified at the top of the file as well (otherwise, an exception
  // will be thrown at some point, since the computational kernel defined in
  // the templated LaplaceOperator class and the information from the finite
  // element read out by MatrixFree will not match). The constructor of the
  // triangulation needs to set an additional flag that tells the grid to
  // conform to the 2:1 cell balance over vertices, which is needed for the
  // convergence of the geometric multigrid routines. For the distributed
  // grid, we also need to specifically enable the multigrid hierarchy.
  template <int dim,int degree_finite_element>
  LaplaceProblem<dim,degree_finite_element>::LaplaceProblem ()
    :
    triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
    fe (degree_finite_element),
    dof_handler (triangulation),
    mapping (std::min(10,degree_finite_element)),
    setup_time(0.),
    pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    // The LaplaceProblem class holds an additional output stream that
    // collects detailed timings about the setup phase. This stream, called
    // time_details, is disabled by default through the @p false argument
    // specified here. For detailed timings, removing the @p false argument
    // prints all the details.
    time_details (std::cout, false &&
                  Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {}



  // @sect4{LaplaceProblem::setup_system}

  // The setup stage is in analogy to step-16 with relevant changes due to the
  // LaplaceOperator class. The first thing to do is to set up the DoFHandler,
  // including the degrees of freedom for the multigrid levels, and to
  // initialize constraints from hanging nodes and homogeneous Dirichlet
  // conditions. Since we intend to use this programs in %parallel with MPI,
  // we need to make sure that the constraints get to know the locally
  // relevant degrees of freedom, otherwise the storage would explode when
  // using more than a few hundred millions of degrees of freedom, see
  // step-40.

  // Once we have created the multigrid dof_handler and the constraints, we
  // can call the reinit function for the global matrix operator as well as
  // each level of the multigrid scheme. The main action is to set up the
  // <code> MatrixFree </code> instance for the problem. The base class of the
  // <code>LaplaceOperator</code> class, MatrixFreeOperators::Base, is
  // initialized with a shared pointer to MatrixFree object. This way, we can
  // simply create it here and then pass it on to the system matrix and level
  // matrices, respectively. For setting up MatrixFree, we need to activate
  // the update flag in the AdditionalData field of MatrixFree that enables
  // the storage of quadrature point coordinates in real space (by default, it
  // only caches data for gradients (inverse transposed Jacobians) and JxW
  // values). Note that if we call the reinit function without specifying the
  // level (i.e., giving <code>level = numbers::invalid_unsigned_int</code>),
  // MatrixFree constructs a loop over the active cells. In this tutorial, we
  // do not use threads in addition to MPI, which is why we explicitly disable
  // it by setting the MatrixFree::AdditionalData::tasks_parallel_scheme to
  // MatrixFree::AdditionalData::none. Finally, the coefficient is evaluated
  // and vectors are initialized as explained above.
  template <int dim,int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::setup_system ()
  {
    Timer time;
    setup_time = 0;

    dof_handler.distribute_dofs (fe);
    dof_handler.distribute_mg_dofs ();
    /*
    Point<dim> downstream;
    downstream[0] = 1;
    for (unsigned int d=1; d<dim; ++d)
      downstream[d] = 1e-5*downstream[d-1];
    for (unsigned int l=0; l<triangulation.n_levels(); ++l)
      DoFRenumbering::downstream(dof_handler, l, downstream, true);
    */
    pcout << "Number of degrees of freedom: "
          << dof_handler.n_dofs() << " = ("
          << ((int)std::pow(dof_handler.n_dofs()*1.0000001, 1./dim)-1)/fe.degree
          << " x " << fe.degree << " + 1)^" << dim
          << std::endl;

    solution.reinit(dof_handler.n_dofs());

    setup_time += time.wall_time();

    pcout << "Total setup time:      " << setup_time
          << "s" << std::endl;
  }



  template<typename VectorType, typename SmootherType>
  class MGCoarseFromSmoother : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseFromSmoother(const SmootherType &mg_smoother,
                         const bool is_empty)
      : smoother(mg_smoother),
        is_empty(is_empty)
    {}

    virtual void operator() (const unsigned int level,
                             VectorType        &dst,
                             const VectorType  &src) const
    {
      if (is_empty)
        return;
      smoother[level].vmult(dst, src);
    }

    const SmootherType &smoother;
    const bool is_empty;
  };



  template <int dim, int fe_degree, typename Number, typename Number2>
  class MultigridSolver
  {
  public:
    MultigridSolver(const DoFHandler<dim>     &dof_handler,
                    const unsigned int         degree_pre,
                    const unsigned int         degree_post,
                    const unsigned int         n_cycles = 1)
      :
      dof_handler(&dof_handler),
      minlevel(0),
      maxlevel(dof_handler.get_triangulation().n_global_levels()-1),
      defect(minlevel, maxlevel),
      solution(minlevel, maxlevel),
      t(minlevel, maxlevel),
      rhs(minlevel, maxlevel),
      residual(minlevel, maxlevel),
      defect2(minlevel, maxlevel),
      matrix(minlevel, maxlevel),
      matrix_dp(minlevel, maxlevel),
      smooth(minlevel, maxlevel),
      coarse (smooth, false),
      degree_pre(degree_pre),
      degree_post(degree_post),
      timings(maxlevel+1),
      n_cycles (n_cycles)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      std::set<types::boundary_id> dirichlet_boundary;
      dirichlet_boundary.insert(0);
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

      MappingQGeneric<dim> mapping(std::min(fe_degree, 10));

      for (unsigned int level=minlevel; level<=maxlevel; ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler, level,
                                                        relevant_dofs);
          AffineConstraints<double> level_constraints;
          level_constraints.reinit(relevant_dofs);
          level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
          level_constraints.close();

          matrix[level].reinit(dof_handler, mg_constrained_dofs, level);
          matrix_dp[level].reinit(dof_handler, mg_constrained_dofs, level);
          defect[level].reinit(matrix[level].n());
          t[level] = defect[level];
          defect2[level] = defect[level];
          solution[level].reinit(matrix[level].n());
          rhs[level] = solution[level];
          residual[level] = solution[level];
        }

      Timer time;

      mg_transfer_no_boundary.build(dof_handler);
      transfer.initialize_constraints(mg_constrained_dofs);
      transfer.build(dof_handler);

      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel+1);
      Solution<dim> analytic_solution;
      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          Quadrature<dim-1> face_quad(dof_handler.get_fe().get_unit_face_support_points());
          FEFaceValues<dim> fe_values(mapping, dof_handler.get_fe(), face_quad,
                                      update_quadrature_points);
          std::vector<types::global_dof_index> face_dof_indices(dof_handler.get_fe().dofs_per_face);

          typename DoFHandler<dim>::cell_iterator
            cell = dof_handler.begin(level),
            endc = dof_handler.end(level);
          for (; cell != endc; ++cell)
            if (cell->level_subdomain_id() != numbers::artificial_subdomain_id)
              for (unsigned int face_no = 0;
                   face_no < GeometryInfo<dim>::faces_per_cell;
                   ++face_no)
                if (cell->at_boundary(face_no))
                  {
                    const typename DoFHandler<dim>::face_iterator
                                           face = cell->face(face_no);
                    face->get_mg_dof_indices(level, face_dof_indices);
                    fe_values.reinit(cell, face_no);
                    for (unsigned int i=0; i<face_dof_indices.size(); ++i)
                      if (dof_handler.locally_owned_mg_dofs(level).is_element(face_dof_indices[i]))
                        {
                          const double value = analytic_solution.value(fe_values.quadrature_point(i));
                          if (value != 0.0)
                            inhomogeneous_bc[level][face_dof_indices[i]] = value;
                        }
                  }

	  matrix_dp[level].constraint_handler.reinit(inhomogeneous_bc[level]);

          typename MatrixFree<dim,Number2>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim,Number2>::AdditionalData::partition_color;
          additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                  update_quadrature_points);
          additional_data.level_mg_handler = level;
	  AffineConstraints<double> level_constraints;
	  level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
	  level_constraints.close();
          std::vector<const AffineConstraints<double>*> constraints(1, &level_constraints);
          std::vector<const DoFHandler<dim>*> dof_handlers(1, &dof_handler);
          std::vector<QGauss<1>> quadratures;
          quadratures.emplace_back(fe_degree+1);

          std::shared_ptr<MatrixFree<dim,Number2> >
            mg_mf_storage_level(new MatrixFree<dim,Number2>());
          mg_mf_storage_level->reinit(mapping, dof_handlers, constraints,
                                      quadratures, additional_data);

          // compute rhs
          Vector<Number2> solution_host(dof_handler.n_dofs(level));
          Vector<Number2> rhs_host(dof_handler.n_dofs(level));
          for (auto &i : inhomogeneous_bc[level])
            if (dof_handler.locally_owned_mg_dofs(level).is_element(i.first))
              solution_host(i.first) = i.second;

	  const std::function< void(const MatrixFree< dim, Number2> &, Vector<Number2> &, const Vector<Number2> &, const std::pair< unsigned int, unsigned int > &)> my_func
	    = [](const MatrixFree<dim,Number2> &data,
		 Vector<double> &rhs_cpu,
		 const Vector<double> &sol_cpu,
		 const std::pair<unsigned int,unsigned int> &cell_range)
	    {
	      RightHandSide<dim> right_hand_side;
	      FEEvaluation<dim,fe_degree,fe_degree+1,1,Number2> phi(data);
	      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
		{
		  phi.reinit(cell);
		  phi.read_dof_values_plain(sol_cpu);
		  phi.evaluate(false, true);
		  for (unsigned int q=0; q<phi.n_q_points; ++q)
		    {
		      Point<dim,VectorizedArray<Number2>> pvec = phi.quadrature_point(q);
		      VectorizedArray<Number2> rhs_val;
		      for (unsigned int v=0; v<VectorizedArray<Number2>::n_array_elements; ++v)
			{
			  Point<dim> p;
			  for (unsigned int d=0; d<dim; ++d)
			    p[d] = pvec[d][v];
			  rhs_val[v] = right_hand_side.value(p);
			}
		      phi.submit_value(rhs_val, q);
		      phi.submit_gradient(-phi.get_gradient(q), q);
		    }
		  phi.integrate_scatter(true, true, rhs_cpu);
		}
	    };
          mg_mf_storage_level->cell_loop
            (my_func, rhs_host, solution_host);
          rhs[level] = rhs_host;
        }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        std::cout << "Time compute rhs:      " << time.wall_time() << std::endl;

      time.restart();
      for (unsigned int level = minlevel; level<=maxlevel; ++level)
      {
        typename SmootherType::AdditionalData smoother_data;
        if (level > minlevel)
          {
            smoother_data.smoothing_range = 24.;
            smoother_data.degree = degree_pre;
            smoother_data.eig_cg_n_iterations = 20;
          }
        else
          {
            smoother_data.smoothing_range = 2e-3;
            smoother_data.degree = numbers::invalid_unsigned_int;
            smoother_data.eig_cg_n_iterations = matrix[minlevel].m();
          }
	/*
	FullMatrix<double> mat(27, 27);
	Vector<Number> v1(matrix[level].m());
	Vector<Number> v2(matrix[level].m());
	for (unsigned int i=0; i<mat.m(); ++i)
	  {
	    v1 = 0;
	    v1(i+98) = 1;
	    defect[level] = v1;
	    matrix[level].vmult(defect2[level], defect[level]);
	    defect2[level].copyToHost(v2);
	    for (unsigned int j=0; j<mat.m(); ++j)
	      mat(j,i) = v2(j+98);
	  }
	defect[level] = 0;
	defect2[level] = 0;
	mat.print_formatted(std::cout, 5, true, 12, "0");
	*/
        matrix[level].compute_diagonal();
        smoother_data.preconditioner = matrix[level].get_matrix_diagonal_inverse();
	//Vector<Number> vec(matrix[level].m());
	//smoother_data.preconditioner->get_vector().copyToHost(vec);
	//vec.print(std::cout);
        smooth[level].initialize(matrix[level], smoother_data);
      }
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        std::cout << "Time initial smoother: " << time.wall_time() << std::endl;

    }

    void solve(const bool do_analyze)
    {
      Timer time;
      defect[minlevel] = rhs[minlevel];
      coarse(minlevel, t[minlevel], defect[minlevel]);
      smooth[minlevel].step(t[minlevel], defect[minlevel]);
      solution[minlevel] = t[minlevel];
      timings[minlevel][0] += time.wall_time();
      timings[minlevel][1] += 2;
      for (unsigned int level=minlevel+1; level<=maxlevel; ++level)
        {
          Timer time;
          matrix_dp[level-1].constraint_handler.apply_boundary_values(solution[level-1]);
          timings[level][3] += time.wall_time();
          //if (level < 4)
          //solution[level-1].print(std::cout);
          time.restart();
          mg_transfer_no_boundary.prolongate(level, solution[level], solution[level-1]);
          timings[level][2] += time.wall_time();

          unsigned int n_iter = (level == maxlevel-1) ? 1 : 1;

          for (unsigned int i=0; i<n_iter; ++i)
            {
              time.restart();
              matrix_dp[level].vmult(residual[level], solution[level]);
              timings[level][0] += time.wall_time();
              time.restart();
              residual[level].sadd(-1., 1., rhs[level]);
              defect[level] = residual[level];
              timings[level][4] += time.wall_time();

              v_cycle(level, n_cycles);

              time.restart();
              residual[level] = defect2[level];
              solution[level] += residual[level];
              timings[level][4] += time.wall_time();
            }
        }
    }

    std::pair<unsigned int, double> solve_cg()
    {
      ReductionControl solver_control(100, 1e-16, 1e-9);
      SolverCG<VectorType2> solver_cg(solver_control);
      solution[maxlevel] = 0;
      solver_cg.solve(matrix_dp[maxlevel], solution[maxlevel], rhs[maxlevel],
                      *this);
      return std::make_pair(solver_control.last_step(),
                            std::pow(solver_control.last_value()/solver_control.initial_value(),
                                     1./solver_control.last_step()));
    }

    void print_wall_times()
    {
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << "Coarse solver " << (int)timings[minlevel][1]
                    << " times: " << timings[minlevel][0]  << " tot prec " << timings[minlevel][2] << std::endl;
          std::cout << "level  smooth_mv  smooth_vec  mg_mv     mg_vec    restrict  prolongate  inhomBC" << std::endl;
          for (unsigned int level=minlevel+1; level<=maxlevel; ++level)
            {
              auto ar = smooth[level].get_timings();
              std::cout << "L"  << std::setw(2) << std::left << level << "    ";
              std::cout << std::setprecision(4)
                        << std::setw(11) << ar[0]
                        << std::setw(12) << ar[1]
                        << std::setw(10) << timings[level][0]
                        << std::setw(10) << timings[level][4]
                        << std::setw(10) << timings[level][1]
                        << std::setw(12) << timings[level][2]
                        << std::setw(10) << timings[level][3]
                        << std::endl;
            }
          std::cout << std::setprecision(5);
        }
      for (unsigned int l=0; l<timings.size(); ++l)
        for (unsigned int j=0; j<timings[l].size(); ++j)
          timings[l][j] = 0.;
    }

    void copy_solution_to_host(Vector<Number2> &dst)
    {
      matrix_dp[maxlevel].constraint_handler.apply_boundary_values(solution[maxlevel]);
      solution[maxlevel].copyToHost(dst);
    }

    void vmult (GpuVector<Number2>       &dst,
                const GpuVector<Number2> &src) const
    {
      Timer time, time1;
      defect[maxlevel] = src;
      timings[maxlevel][4] += time.wall_time();
      v_cycle(maxlevel, 1);
      time.restart();
      dst = defect2[maxlevel];
      timings[maxlevel][4] += time.wall_time();
      timings[minlevel][2] += time1.wall_time();
    }

    void do_matvec()
    {
      matrix_dp[maxlevel].vmult(residual[maxlevel], solution[maxlevel]);
    }

    void do_matvec_smoother()
    {
      matrix[maxlevel].vmult(defect2[maxlevel], defect[maxlevel]);
    }

  private:
    /**
     * A pointer to the DoFHandler object
     */
    const SmartPointer<const DoFHandler<dim> > dof_handler;

    std::vector<std::map<types::global_dof_index,Number2>> inhomogeneous_bc;

    MGConstrainedDoFs mg_constrained_dofs;

    MGTransferMatrixFreeGpu<dim,Number2> mg_transfer_no_boundary;
    MGTransferMatrixFreeGpu<dim,Number> transfer;

    typedef GpuVector<Number>  VectorType;
    typedef GpuVector<Number2> VectorType2;
    typedef PreconditionChebyshev<LaplaceOperatorGpu<dim,fe_degree,Number>,VectorType> SmootherType;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType> defect;

    /**
     * The solution update after the multigrid step.
     */
    mutable MGLevelObject<VectorType2> solution;
    mutable MGLevelObject<VectorType2> residual;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType> t;

    /**
     * Auxiliary vector if more than 1 cycle is needed
     */
    mutable MGLevelObject<VectorType> defect2;
    mutable MGLevelObject<VectorType2> rhs;

    /**
     * The matrix for each level.
     */
    MGLevelObject<LaplaceOperatorGpu<dim,fe_degree,Number> > matrix;
    MGLevelObject<LaplaceOperatorGpu<dim,fe_degree,Number2> > matrix_dp;

    /**
     * The matrix for each level.
     */
    MGCoarseFromSmoother<VectorType,MGLevelObject<SmootherType>> coarse;

    /**
     * The smoothing object.
     */
    MGLevelObject<SmootherType> smooth;

    const unsigned int degree_pre;
    const unsigned int degree_post;

    const unsigned int n_cycles;

    mutable std::vector<std::array<double,5>> timings;

    /**
     * Implements the v-cycle
     */
    void v_cycle(const unsigned int level,
                 const unsigned int my_n_cycles) const
    {
      if (level==minlevel)
        {
          Timer time;
          (coarse)(level, defect2[level], defect[level]);
          timings[level][0] += time.wall_time();
          timings[level][1] += 1;
          return;
        }

      Timer time;
      for (unsigned int c=0; c<my_n_cycles; ++c)
        {
          (smooth)[level].set_degree(degree_pre);

          if (c==0)
            (smooth)[level].vmult(defect2[level], defect[level]);
          else
            (smooth)[level].step(defect2[level], defect[level]);

          time.restart();
          (matrix)[level].vmult/*_interface_down*/(t[level], defect2[level]);
          timings[level][0] += time.wall_time();
          time.restart();
          t[level].sadd(-1.0, 1.0, defect[level]);
          timings[level][4] += time.wall_time();
          time.restart();

          defect[level-1] = 0;
          transfer.restrict_and_add(level, defect[level-1], t[level]);
          timings[level][1] += time.wall_time();
          //(*matrix)[level].vmult_subtract_edge_down(defect[level-1], solution[level]);

          v_cycle(level-1, 1);

          time.restart();
          transfer.prolongate(level, t[level], defect2[level-1]);
          timings[level][2] += time.wall_time();
          time.restart();
          defect2[level] += t[level];
          timings[level][4] += time.wall_time();
          //(*matrix)[level].vmult_add/*_interface_up*/(defect[level], solution[level]);
          //(*matrix)[level].vmult_add_edge_up(defect[level], solution[level-1]);
          (smooth)[level].set_degree(degree_post);
          (smooth)[level].step(defect2[level], defect [level]);
        }
    }
  };



  template <int dim,int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::solve (const unsigned int n_mg_cycles,
                                                         const unsigned int n_pre_smooth,
                                                         const unsigned int n_post_smooth)
  {
    MultigridSolver<dim, degree_finite_element, level_number, full_number>
      solver(dof_handler, n_pre_smooth, n_post_smooth, n_mg_cycles);

    Timer time;

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory =
      Utilities::MPI::min_max_avg (stats.VmRSS/1024., MPI_COMM_WORLD);

    pcout << "Memory stats [MB]: " << memory.min
          << " [p" << memory.min_index << "] "
          << memory.avg << " " << memory.max
          << " [p" << memory.max_index << "]"
          << std::endl;

    double best_time = 1e10, tot_time = 0;
    for (unsigned int i=0; i<15; ++i)
      {
        cudaDeviceSynchronize();
        time.restart();
        solver.solve(false);
        cudaDeviceSynchronize();
        best_time = std::min(time.wall_time(), best_time);
        tot_time += time.wall_time();
        pcout << "Time solve   (CPU/wall)    " << time.cpu_time() << "s/"
              << time.wall_time() << "s\n";
      }
    Utilities::MPI::MinMaxAvg stat =
      Utilities::MPI::min_max_avg (tot_time, MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "All solver time " << stat.min << " [p" << stat.min_index << "] "
                  << stat.avg << " " << stat.max << " [p" << stat.max_index << "]"
                  << std::endl;
    solver.print_wall_times();

    solver.copy_solution_to_host(solution);
    MappingQGeneric<dim> mapping(fe.degree);
    Vector<float> error_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference (mapping,
                                       dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       error_per_cell,
                                       QGauss<dim>(fe.degree+1),
                                       VectorTools::L2_norm);
    const double l2_error = sqrt(error_per_cell.norm_sqr());

    cudaDeviceSynchronize();
    time.restart();
    auto cg_details = solver.solve_cg();
    cudaDeviceSynchronize();
    const double time_cg = time.wall_time();
    solver.print_wall_times();
    solver.copy_solution_to_host(solution);
    VectorTools::integrate_difference (mapping,
                                       dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       error_per_cell,
                                       QGauss<dim>(fe.degree+1),
                                       VectorTools::L2_norm);
    const double l2_error_cg = sqrt(error_per_cell.norm_sqr());

    double best_mv = 1e10;
    for (unsigned int i=0; i<5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 2000 : 250;
        cudaDeviceSynchronize();
        time.restart();
        for (unsigned int i=0; i<n_mv; ++i)
          solver.do_matvec();
        cudaDeviceSynchronize();
        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg (time.wall_time()/n_mv, MPI_COMM_WORLD);
        best_mv = std::min(best_mv, stat.max);
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "matvec time dp " << stat.min << " [p" << stat.min_index << "] "
                    << stat.avg << " " << stat.max << " [p" << stat.max_index << "]"
                    << " DoFs/s: " << dof_handler.n_dofs() / stat.max
                    << std::endl;
      }
    double best_mvs = 1e10;
    for (unsigned int i=0; i<5; ++i)
      {
        const unsigned int n_mv = dof_handler.n_dofs() < 10000000 ? 2000 : 250;
        cudaDeviceSynchronize();
        time.restart();
        for (unsigned int i=0; i<n_mv; ++i)
          solver.do_matvec_smoother();
        cudaDeviceSynchronize();
        Utilities::MPI::MinMaxAvg stat =
          Utilities::MPI::min_max_avg (time.wall_time()/n_mv, MPI_COMM_WORLD);
        best_mvs = std::min(best_mvs, stat.max);
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "Best timings for " << dof_handler.n_dofs() << "   mv "
                << best_mv << "    mv smooth " << best_mvs
                << "   mg " << best_time << std::endl;

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "L2 error at " << dof_handler.n_dofs() << "  "
                << l2_error << "  with CG " << l2_error_cg << std::endl;

    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());
    convergence_table.add_value("L2error", l2_error);
    convergence_table.add_value("mv_outer", best_mv);
    convergence_table.add_value("mv_inner", best_mvs);
    convergence_table.add_value("fmg_cycle", best_time);
    convergence_table.add_value("L2error_cg", l2_error_cg);
    convergence_table.add_value("cg_time", time_cg);
    convergence_table.add_value("cg_its", cg_details.first);
    convergence_table.add_value("cg_reduction", cg_details.second);
  }



  // @sect4{LaplaceProblem::output_results}

  // Here is the data output, which is a simplified version of step-5. We use
  // the standard VTU (= compressed VTK) output for each grid produced in the
  // refinement process. We disable the output when the mesh gets too
  // large. Note that a variant of program has been run on hundreds of
  // thousands MPI ranks with as many as 100 billion grid cells, which is not
  // directly accessible to classical visualization tools.
  template <int dim,int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::output_results (const unsigned int cycle) const
  {
    //if (triangulation.n_global_active_cells() > 10000)
      return;

    DataOut<dim> data_out;

    Vector<double> mysol(dof_handler.n_dofs());
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    Vector<double> vec2, vec3;
    vec2.reinit(solution);
    VectorTools::interpolate(mapping, dof_handler, Solution<dim>(), vec2);
    data_out.add_data_vector (vec2, "exact");
    vec3 = vec2;
    vec3 -= solution;
    data_out.add_data_vector (vec3, "error");
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);
    data_out.build_patches (fe.degree);

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


template <int dim>
class MyManifold : public dealii::ChartManifold<dim,dim,dim>
{
public:
  MyManifold() : factor(0.1) {}

  virtual std::unique_ptr<dealii::Manifold<dim> > clone () const
  {
    return dealii::std_cxx14::make_unique<MyManifold<dim>>();
  }

  virtual dealii::Point<dim> push_forward(const dealii::Point<dim> &p) const
  {
    double sinval = factor;
    for (unsigned int d=0; d<dim; ++d)
      sinval *= std::sin(dealii::numbers::PI*p[d]);
    dealii::Point<dim> out;
    for (unsigned int d=0; d<dim; ++d)
      out[d] = p[d] + sinval;
    return out;
  }

  virtual dealii::Point<dim> pull_back(const dealii::Point<dim> &p) const
  {
    dealii::Point<dim> x = p;
    dealii::Point<dim> one;
    for (unsigned int d=0; d<dim; ++d)
      one(d) = 1.;

    dealii::Tensor<1,dim> sinvals;
    for (unsigned int d=0; d<dim; ++d)
      sinvals[d] = std::sin(dealii::numbers::PI*x(d));

    double sinval = factor;
    for (unsigned int d=0; d<dim; ++d)
      sinval *= sinvals[d];
    dealii::Tensor<1,dim> residual = p - x - sinval*one;
    unsigned int its = 0;
    while (residual.norm() > 1e-12 && its < 100)
      {
        dealii::Tensor<2,dim> jacobian;
        for (unsigned int d=0; d<dim; ++d)
          jacobian[d][d] = 1.;
        for (unsigned int d=0; d<dim; ++d)
          {
            double sinval_der = factor * dealii::numbers::PI * std::cos(dealii::numbers::PI*x(d));
            for (unsigned int e=0; e<dim; ++e)
              if (e!=d)
                sinval_der *= sinvals[e];
            for (unsigned int e=0; e<dim; ++e)
              jacobian[e][d] += sinval_der;
          }

        x += dealii::invert(jacobian) * residual;

        for (unsigned int d=0; d<dim; ++d)
          sinvals[d] = std::sin(dealii::numbers::PI*x(d));

        sinval = factor;
        for (unsigned int d=0; d<dim; ++d)
          sinval *= sinvals[d];
        residual = p - x - sinval*one;
        ++its;
      }
    AssertThrow (residual.norm() < 1e-12,
                 dealii::ExcMessage("Newton for point did not converge."));
    return x;
  }

private:
  const double factor;
};

  // @sect4{LaplaceProblem::run}

  // The function that runs the program is very similar to the one in
  // step-16. We do few refinement steps in 3D compared to 2D, but that's
  // it.
  template <int dim,int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::run (const unsigned int max_size,
                                                       const unsigned int n_mg_cycles,
                                                       const unsigned int n_pre_smooth,
                                                       const unsigned int n_post_smooth)
  {
    pcout << "Testing " << fe.get_name() << std::endl;
    for (unsigned int cycle=8; cycle<35; ++cycle)
      {
	triangulation.clear();
	pcout << "Cycle " << cycle << std::endl;

	const unsigned int n_refine = cycle/3;
	const unsigned int remainder = cycle%3;
	Point<dim> p1;
	for (unsigned int d=0; d<dim; ++d)
	  p1[d] = -1;
	Point<dim> p2;
	for (unsigned int d=0; d<remainder; ++d)
	  p2[d] = 2.8;
	for (unsigned int d=remainder; d<dim; ++d)
	  p2[d] = 1;
	std::vector<unsigned int> subdivisions(dim, 1);
	for (unsigned int d=0; d<remainder; ++d)
	  subdivisions[d] = 2;
	GridGenerator::subdivided_hyper_rectangle(triangulation, subdivisions, p1, p2);

#ifdef CURVED_GRID
        MyManifold<dim> manifold;
        GridTools::transform(std::bind(&MyManifold<dim>::push_forward, manifold,
                                       std::placeholders::_1),
                             triangulation);
        triangulation.set_all_manifold_ids(1);
        triangulation.set_manifold(1, manifold);
#endif

        triangulation.refine_global(n_refine);

        setup_system ();
        if (dof_handler.n_dofs() > max_size)
          {
            pcout << "Max size reached, terminating." << std::endl;
            pcout << std::endl;
            break;
          }


        solve (n_mg_cycles, n_pre_smooth, n_post_smooth);
        output_results (cycle);
        pcout << std::endl;
      };

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        convergence_table.set_scientific("L2error", true);
        convergence_table.set_precision("L2error", 3);
        convergence_table.evaluate_convergence_rates("L2error", "cells",
                                                     ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("mv_outer", true);
        convergence_table.set_precision("mv_outer", 3);
        convergence_table.set_scientific("mv_inner", true);
        convergence_table.set_precision("mv_inner", 3);
        convergence_table.set_scientific("fmg_cycle", true);
        convergence_table.set_precision("fmg_cycle", 3);

        convergence_table.set_scientific("reduction", true);
        convergence_table.set_precision("reduction", 3);
        convergence_table.set_scientific("L2error_cg", true);
        convergence_table.set_precision("L2error_cg", 3);
        convergence_table.evaluate_convergence_rates("L2error_cg", "cells",
                                                     ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("cg_reduction", true);
        convergence_table.set_precision("cg_reduction", 3);
        convergence_table.set_scientific("cg_time", true);
        convergence_table.set_precision("cg_time", 3);

        convergence_table.write_text(std::cout);

        std::cout << std::endl << std::endl;
      }
  }



  template <int dim, int min_degree, int max_degree>
  class LaplaceRunTime {
  public:
    LaplaceRunTime(const unsigned int target_degree,
                   const unsigned int max_size,
                   const unsigned int n_mg_cycles,
                   const unsigned int n_pre_smooth,
                   const unsigned int n_post_smooth)
    {
      if (min_degree>max_degree)
        return;
      if (min_degree == target_degree)
        {
          LaplaceProblem<dim,min_degree> laplace_problem;
          laplace_problem.run(max_size, n_mg_cycles, n_pre_smooth, n_post_smooth);
        }
      LaplaceRunTime<dim,(min_degree<=max_degree?(min_degree+1):min_degree),max_degree>
                     m(target_degree, max_size, n_mg_cycles, n_pre_smooth, n_post_smooth);
    }
  };
}




// @sect3{The <code>main</code> function}

// Apart from the fact that we set up the MPI framework according to step-40,
// there are no surprises in the main function.
int main (int argc, char *argv[])
{
  try
    {
      using namespace Step37;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);


      unsigned int degree = 4;
      unsigned int maxsize = numbers::invalid_unsigned_int;
      unsigned int n_mg_cycles = 1;
      unsigned int n_pre_smooth = 3;
      unsigned int n_post_smooth = 3;
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

      LaplaceRunTime<dimension,minimal_degree,maximal_degree> run(degree, maxsize, n_mg_cycles,
                                                                  n_pre_smooth, n_post_smooth);
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
