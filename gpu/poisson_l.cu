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
 * This program solves the Poisson equation on an L-shaped domain with
 * adaptive mesh refinement. The linear system is solved with the conjugate
 * gradient method preconditioned by a geometric multigrid V-cycle.
 */


#include <deal.II/base/subscriptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/convergence_table.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/multigrid.h>

#include <fstream>
#include <sstream>

// we use adaptive refinement, so need hanging node support
#define MATRIX_FREE_HANGING_NODES
// only use cartesian mesh, so can enable Jacobian optimization
#define MATRIX_FREE_UNIFORM_MESH

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/gpu_array.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/mg_transfer_matrix_free_gpu.h"
#include "matrix_free_gpu/constraint_handler_gpu.h"
#include "matrix_free_gpu/matrix_free_gpu.h"
#include "matrix_free_gpu/fee_gpu.cuh"


#define SAVE_MESH

// #define LAST_CYCLE
// #define LOAD_MESH

namespace Step37
{
  using namespace dealii;


#ifdef SPACE_DIMENSION
  const unsigned int dimension = SPACE_DIMENSION;
#else
  const unsigned int dimension = 2;
#endif

#ifdef USE_3D_HYPER_L
  const bool use_3d_l = true;
#else
  const bool use_3d_l = false;
#endif

#ifdef USE_JACOBI
  const bool use_jacobi = true;
#else
  const bool use_jacobi = false;
#endif

  typedef double number;
  typedef float level_number;


  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution() : Function<dim> (1) {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const
    {
      if (use_3d_l == false)
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
      if (use_3d_l == false)
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
  };

  template <>
  class Solution<2> : public Functions::LSingularityFunction
  {
  public:
    Solution() : Functions::LSingularityFunction() {}
  };


  //---------------------------------------------------------------------------
  // operator
  //---------------------------------------------------------------------------

  template <int dim, int fe_degree,typename Number>
  class LaplaceOperator : public Subscriptor
  {
  public:
    typedef Number value_type;
    typedef GpuVector<Number> VectorType;
    typedef unsigned int size_type;

    LaplaceOperator ();

    void clear();

    void initialize (std::shared_ptr<const MatrixFreeGpu<dim,Number> >   matrix_free,
                     const ConstraintMatrix                              &constraints);


    void initialize (std::shared_ptr<const MatrixFreeGpu<dim,Number>>   matrix_free,
                     const MGConstrainedDoFs                            &mg_constrained_dofs,
                     const unsigned int                                 level);


    unsigned int m () const { return data->n_dofs; }
    unsigned int n () const { return data->n_dofs; }


    // we cannot access matrix elements of a matrix free operator directly.
    Number el (const unsigned int row,
               const unsigned int col) const
    {
      ExcNotImplemented();
      return -1000000000000000000;
    }

    void vmult (VectorType &dst,
                const VectorType &src) const;
    void Tvmult (VectorType &dst,
                 const VectorType &src) const;
    void vmult_add (VectorType &dst,
                    const VectorType &src) const;
    void Tvmult_add (VectorType &dst,
                     const VectorType &src) const;
    void vmult_interface_down(VectorType       &dst,
                              const VectorType &src) const;
    void vmult_interface_up(VectorType       &dst,
                            const VectorType &src) const;

    // diagonal for preconditioning
    void compute_diagonal ();

    void compute_residual (VectorType       &dst,
                           const VectorType &src) const;

    const std::shared_ptr<DiagonalMatrix<VectorType>> get_matrix_diagonal_inverse () const;


    void initialize_dof_vector(VectorType &v) const;

    std::size_t memory_consumption () const;


  private:
    unsigned int                level;

    std::shared_ptr<const MatrixFreeGpu<dim,Number>>    data;

    std::shared_ptr<DiagonalMatrix<VectorType>>         inverse_diagonal_matrix;
    bool                                                diagonal_is_available;

    mutable ConstraintHandlerGpu<Number> constraint_handler;
  };



  template <int dim, int fe_degree, typename Number>
  LaplaceOperator<dim,fe_degree,Number>::LaplaceOperator ()
    :
    Subscriptor()
  {
  }



  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::clear ()
  {
    data.reset();
    diagonal_is_available = false;
    inverse_diagonal_matrix.reset();
  }


  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::initialize (std::shared_ptr<const MatrixFreeGpu<dim,Number> > data_in,
                                                     const ConstraintMatrix                            &constraints)
  {
    this->level = numbers::invalid_unsigned_int;

    // this constructor is used for the non-Multigrid case, i.e. when this matrix
    // is the global system matrix

    data = data_in;

    constraint_handler.reinit(constraints,data->n_dofs);
  }


  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::initialize (std::shared_ptr<const MatrixFreeGpu<dim,Number> > data_in,
                                                     const MGConstrainedDoFs                           &mg_constrained_dofs,
                                                     const unsigned int                                level)
  {
    // Multigrid-case constructor, i.e. this matrix is a level-local matrix
    this->level = level;

    data = data_in;

    constraint_handler.reinit(mg_constrained_dofs,
                              level);
  }



  // multiplication/application functions (symmetric operator -> vmult == Tvmult)

  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::vmult (VectorType       &dst,
                                                   const VectorType &src) const
  {
    dst = 0.0;
    vmult_add (dst, src);
  }



  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::Tvmult (VectorType       &dst,
                                                    const VectorType &src) const
  {
    dst = 0.0;
    vmult_add (dst,src);
  }



  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::Tvmult_add (VectorType       &dst,
                                                        const VectorType &src) const
  {
    vmult_add (dst,src);
  }

  // This is the struct we pass to matrix-free for evaluation on each cell
  template <int dim, int fe_degree, typename Number>
  struct LocalOperator {
    static const unsigned int n_dofs_1d = fe_degree+1;
    static const unsigned int n_local_dofs = ipow<fe_degree+1,dim>::val;
    static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

    // what to do on each quadrature point
    template <typename FEE>
    __device__ inline void quad_operation(FEE *phi, const unsigned int q) const
    {
      phi->submit_gradient (phi->get_gradient(q), q);
    }

    // what to do fore each cell
    __device__ void cell_apply (Number                          *dst,
                                const Number                    *src,
                                const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                                const unsigned int cell,
                                SharedData<dim,Number> *shdata) const
    {
      FEEvaluationGpu<dim,fe_degree,Number> phi (cell, gpu_data, shdata);

      phi.read_dof_values(src);

      phi.evaluate (false,true);

      // apply the local operation above
      phi.apply_quad_point_operations(this);

      phi.integrate (false,true);

      phi.distribute_local_to_global (dst);
    }
  };






  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::vmult_add (VectorType       &dst,
                                                       const VectorType &src) const
  {
    // save possibly non-zero values of Dirichlet and hanging-node values on input
    // and output, and set input values to zero to avoid polluting output.
    constraint_handler.save_constrained_values(dst, const_cast<GpuVector<Number>&>(src));

    // apply laplace operator
    LocalOperator<dim,fe_degree,Number> loc_op;
    data->cell_loop (dst,src,loc_op);

    // overwrite Dirichlet values in output with correct values, and reset input
    // to possibly non-zero values.
    constraint_handler.load_and_add_constrained_values(dst, const_cast<GpuVector<Number>&>(src));
  }


  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::
  vmult_interface_down(VectorType       &dst,
                       const VectorType &src) const
  {
    // set zero Dirichlet values on the refinement edges of the input vector (and
    // remember the src values because we need to reset them at the end).
    // since also the real boundary DoFs should be zeroed out, we do everything at once
    constraint_handler.save_constrained_values(const_cast<VectorType&>(src));

    // use temporary destination, is all zero here
    VectorType tmp_dst(dst.size());

    // apply laplace operator
    LocalOperator<dim,fe_degree,Number> loc_op;
    data->cell_loop (tmp_dst,src,loc_op);

    // now zero out everything except the values at the refinement edges,
    dst = 0.0;
    constraint_handler.copy_edge_values(dst,tmp_dst);

    // and restore the src values
    constraint_handler.load_constrained_values(const_cast<VectorType&>(src));
  }

  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::
  vmult_interface_up(VectorType       &dst,
                     const VectorType &src) const
  {
    // printf("calling vmult_if_up... (level=%d)\n",level);
    dst = 0;

    // only use values at refinement edges
    VectorType src_cpy (src.size());
    constraint_handler.copy_edge_values(src_cpy,src);

    // apply laplace operator
    LocalOperator<dim,fe_degree,Number> loc_op;
    data->cell_loop (dst,src_cpy,loc_op);

    // zero out edge values.
    // since boundary values should also be removed, do both at once.
    constraint_handler.set_constrained_values(dst,0.0);
  }


  template <int dim, int fe_degree, typename Number>
  struct ResidualLocalOperator {
    static const unsigned int n_dofs_1d = fe_degree+1;
    static const unsigned int n_local_dofs = ipow<fe_degree+1,dim>::val;
    static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

    // what to do on each quadrature point
    template <typename FEE>
    __device__ inline void quad_operation(FEE *phi, const unsigned int q) const
    {
      phi->submit_gradient (-phi->get_gradient(q), q);
      if (dim==3 && use_3d_l)
        phi->submit_value (1.0, q);
    }

    // what to do fore each cell
    __device__ void cell_apply (Number                          *dst,
                                const Number                    *src,
                                const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                                const unsigned int cell,
                                SharedData<dim,Number> *shdata) const
    {
      FEEvaluationGpu<dim,fe_degree,Number> phi (cell, gpu_data, shdata);

      phi.read_dof_values(src);

      phi.evaluate (false,true);

      // apply the local operation above
      phi.apply_quad_point_operations(this);

      phi.integrate(dim==3 && use_3d_l, true);

      phi.distribute_local_to_global (dst);
    }
  };

  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::
  compute_residual (VectorType       &dst,
                    const VectorType &src) const
  {
    dst = 0;
    // leave constrained values as-is (we want the boundary values, and
    // hanging-node are never read anyways)

    ResidualLocalOperator<dim,fe_degree,Number> res_loc_op;
    data->cell_loop (dst,src,res_loc_op);

    // Set all constrained values to 0 (including hanging nodes, since we don't
    // use them later anyways)

    constraint_handler.set_constrained_values(dst,0);
  }



  template <int dim, int fe_degree, typename Number>
  struct DiagonalLocalOperator {
    static const unsigned int n_dofs_1d = fe_degree+1;
    static const unsigned int n_q_points_1d = fe_degree+1;
    static const unsigned int n_local_dofs = ipow<n_dofs_1d,dim>::val;
    static const unsigned int n_q_points = ipow<n_q_points_1d,dim>::val;

    // what to do fore each cell
    __device__ void cell_apply (Number                          *dst,
                                const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                                const unsigned int cell,
                                SharedData<dim,Number> *shdata) const
    {
      FEEvaluationGpu<dim,fe_degree,Number> phi (cell, gpu_data, shdata);

      Number my_diagonal = 0.0;

      const unsigned int tid = (dim==1 ? threadIdx.x%n_q_points_1d :
                                dim==2 ? threadIdx.x%n_q_points_1d + n_q_points_1d*threadIdx.y :
                                threadIdx.x%n_q_points_1d + n_q_points_1d*(threadIdx.y + n_q_points_1d*threadIdx.z));

      for (unsigned int i=0; i<n_local_dofs; ++i)
      {
        // set to unit vector
        phi.submit_dof_value(i==tid?1.0:0.0, tid);

        __syncthreads();
        phi.evaluate (false, true);

        phi.submit_gradient (phi.get_gradient(tid), tid);
        __syncthreads();
        phi.integrate (false, true);

        if(tid==i)
          my_diagonal = phi.get_value(tid);
      }
      __syncthreads();

      phi.submit_dof_value(my_diagonal, tid);

      phi.distribute_local_to_global (dst);
    }
  };




  // set diagonal (and set values correponding to constrained DoFs to 1)
  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::compute_diagonal()
  {
    inverse_diagonal_matrix.reset(new DiagonalMatrix<VectorType>());

    VectorType &inv_diag = inverse_diagonal_matrix->get_vector();

    inv_diag.reinit(m());

    DiagonalLocalOperator<dim,fe_degree,Number> diag_loc_op;
    data->cell_loop (inv_diag,diag_loc_op);

    constraint_handler.set_constrained_values(inv_diag,1.0);

    inv_diag.invert();

    diagonal_is_available = true;
  }




  template <int dim, int fe_degree, typename Number>
  const std::shared_ptr<DiagonalMatrix<GpuVector<Number>>>
  LaplaceOperator<dim,fe_degree,Number>::get_matrix_diagonal_inverse() const
  {
    Assert (diagonal_is_available == true, ExcNotInitialized());
    return inverse_diagonal_matrix;
  }



  template <int dim, int fe_degree, typename Number>
  void LaplaceOperator<dim,fe_degree,Number>::initialize_dof_vector(VectorType &v) const
  {
    v.reinit (data->n_dofs);
  }

  template <int dim, int fe_degree, typename Number>
  std::size_t
  LaplaceOperator<dim,fe_degree,Number>::memory_consumption () const
  {
    std::size_t bytes = (data->memory_consumption () +
                         MemoryConsumption::memory_consumption(constraint_handler) +
                         MemoryConsumption::memory_consumption(inverse_diagonal_matrix) +
                         MemoryConsumption::memory_consumption(diagonal_is_available));


    return bytes;
  }


  template <int dim,int degree_finite_element>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    void run ();

  private:
    void setup_system ();
    std::pair<unsigned int,std::pair<double,double> > solve (const bool first_time);
    void output_results (const unsigned int cycle) const;
    void interpolate_boundary_values();

    Triangulation<dim>                         triangulation;

    MappingQGeneric<dim>                       mapping;
    FE_Q<dim>                                  fe;
    DoFHandler<dim>                            dof_handler;

    ConstraintMatrix                           constraints;
    std::shared_ptr<MatrixFreeGpu<dim,number> > system_matrix_free;
    typedef LaplaceOperator<dim,degree_finite_element,number> SystemMatrixType;
    SystemMatrixType                           system_matrix;

    MGConstrainedDoFs                          mg_constrained_dofs;
    MGTransferMatrixFreeGpu<dim,level_number>  mg_transfer;
    typedef LaplaceOperator<dim,degree_finite_element,level_number>  LevelMatrixType;
    MGLevelObject<LevelMatrixType>             mg_matrices;

    GpuVector<number> solution;
    GpuVector<number> solution_update;
    GpuVector<number> system_rhs;
    Vector<number>    solution_host;

    double                                     setup_time;
    ConditionalOStream                         time_details;
  };



  template <int dim,int degree_finite_element>
  LaplaceProblem<dim,degree_finite_element>::LaplaceProblem ()
    :
    triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
    mapping (degree_finite_element),
    fe (degree_finite_element),
    dof_handler (triangulation),
    time_details (std::cout,false)
  {}



  template <int dim,int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::setup_system ()
  {
    Timer time;
    time.start ();
    setup_time = 0;

    system_matrix_free.reset(new MatrixFreeGpu<dim,number>());
    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs (fe);
    dof_handler.distribute_mg_dofs (fe);

    std::cout << "Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << std::endl;

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints);
    constraints.close();

    setup_time += time.wall_time();
    time_details << "Distribute DoFs & B.C.     (CPU/wall) "
                 << time() << "s/" << time.wall_time() << "s" << std::endl;
    time.restart();

    {
      typename MatrixFreeGpu<dim,number>::AdditionalData additional_data;
#ifdef MATRIX_FREE_COLOR
      additional_data.use_coloring = true;
#else
      additional_data.use_coloring = false;
#endif
      additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                              update_quadrature_points);

      system_matrix_free->reinit (mapping, dof_handler, constraints,
                                  QGauss<1>(fe.degree+1), additional_data);
      system_matrix.initialize (system_matrix_free,constraints);
    }

    solution_host.reinit(system_matrix.m());

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(solution_update);
    system_matrix.initialize_dof_vector(system_rhs);

    setup_time += time.wall_time();
    time_details << "Setup matrix-free system   (CPU/wall) "
                 << time() << "s/" << time.wall_time() << "s" << std::endl;
    time.restart();

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels-1);

    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

    for (unsigned int level=0; level<nlevels; ++level)
    {

      typename MatrixFreeGpu<dim,level_number>::AdditionalData additional_data;

#ifdef MATRIX_FREE_COLOR
      additional_data.use_coloring = true;
#else
      additional_data.use_coloring = false;
#endif

      additional_data.parallelization_scheme = MatrixFreeGpu<dim,level_number>::scheme_par_in_elem;
      additional_data.level_mg_handler = level;
      additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                              update_quadrature_points);
      std::shared_ptr<MatrixFreeGpu<dim,level_number> >
        mg_matrix_free_level(new MatrixFreeGpu<dim,level_number>());

      // disable hanging nodes by submitting empty ConstraintMatrix (constraint
      // handler inside Operator takes care of BC)
      mg_matrix_free_level->reinit(mapping, dof_handler, ConstraintMatrix(),
                                   QGauss<1>(fe.degree+1), additional_data);

      mg_matrices[level].initialize(mg_matrix_free_level, mg_constrained_dofs,
                                    level);
    }
    setup_time += time.wall_time();
    time_details << "Setup matrix-free levels   (CPU/wall) "
                 << time() << "s/" << time.wall_time() << "s" << std::endl;

    time.restart();

    mg_transfer.clear();
    mg_transfer.initialize_constraints(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    setup_time += time.wall_time();
    time_details << "MG build transfer time     (CPU/wall) " << time()
                 << "s/" << time.wall_time() << "s\n";
    std::cout << "Total setup time               (wall) " << setup_time
          << "s\n";
  }



  template <int dim,int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::interpolate_boundary_values()
  {
    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(mapping, dof_handler, 0,
                                             Solution<dim>(),
                                             boundary_values);
    for (typename std::map<types::global_dof_index, double>::iterator it = boundary_values.begin();
         it != boundary_values.end(); ++it)
      solution_host(it->first) = it->second;

    // compute hanging node values close to the boundary
    ConstraintMatrix hn_constraints;
    DoFTools::make_hanging_node_constraints(dof_handler,hn_constraints);
    hn_constraints.close();
    hn_constraints.distribute(solution_host);

    solution = solution_host;
  }



  template <int dim,int degree_finite_element>
  std::pair<unsigned int,std::pair<double,double> > LaplaceProblem<dim,degree_finite_element>::solve (const bool /*first_time*/)
  {
    Timer time;
    typedef PreconditionChebyshev<LevelMatrixType, GpuVector<level_number> > SmootherType;
    mg::SmootherRelaxation<SmootherType, GpuVector<level_number> >
      mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
    {
      if (level > 0)
      {
        smoother_data[level].smoothing_range = 25.;
        if (use_jacobi)
          smoother_data[level].degree = 0;
        else
          smoother_data[level].degree = 5;
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

    MGCoarseGridApplySmoother<GpuVector<level_number> > mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<GpuVector<level_number> > mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType> > mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
      mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<GpuVector<level_number> > mg_interface(mg_interface_matrices);

    Multigrid<GpuVector<level_number> > mg(mg_matrix,
                                            mg_coarse,
                                            mg_transfer,
                                            mg_smoother,
                                            mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, GpuVector<level_number>,
      MGTransferMatrixFreeGpu<dim,level_number> >
      preconditioner(dof_handler, mg, mg_transfer);

    ReductionControl solver_control (system_matrix.m(), 1e-13, 1e-9);
    SolverCG<GpuVector<number> > cg (solver_control);
    std::cout << "MG build smoother time     (CPU/wall) " << time()
          << "s/" << time.wall_time() << "s\n";

    system_matrix.compute_residual(system_rhs, solution);

    time.reset();
    cudaDeviceSynchronize();
    time.start();
    solution_update = 0.0;
    cg.solve (system_matrix, solution_update, system_rhs,
              preconditioner);

    cudaDeviceSynchronize();
    time.stop();
    std::pair<unsigned int,std::pair<double,double> > stats(solver_control.last_step(),
                                                            std::pair<double,double>(time.wall_time(),0));
    time.reset();
    time.start();
    solution_update = 0.0;
    cg.solve (system_matrix, solution_update, system_rhs,
              preconditioner);

    cudaDeviceSynchronize();
    time.stop();
    stats.second.second = time.wall_time();


    std::cout << "Time solve ("
              << stats.first
              << " iterations)  (CPU/wall) " << time() << "s/"
              << stats.second.second << "s  convergence rate "
          << std::pow(solver_control.last_value()/solver_control.initial_value(), 1./stats.first) << std::endl;

    Vector<number> solution_update_host = solution_update.toVector();
    constraints.distribute(solution_update_host);
    solution_host += solution_update_host;

    return stats;
  }


  template <int dim, int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::output_results (const unsigned int cycle) const
  {
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    // solution_host.update_ghost_values();
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution_host, "solution");
    data_out.build_patches (mapping, 1, DataOut<dim>::curved_inner_cells);

    std::ostringstream filename;
    filename << "solution-"
             << cycle
             << ".vtu";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);

  }


  void create_tria (Triangulation<2> &tria)
  {
    GridGenerator::hyper_L (tria);
  }

  void create_tria (Triangulation<3> &tria)
  {
    if (use_3d_l)
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


  template <int dim,int degree_finite_element>
  void LaplaceProblem<dim,degree_finite_element>::run ()
  {
    ConvergenceTable     convergence_table;

#ifdef SAVE_MESH
    std::ofstream history("mesh.history");
#endif

#ifdef LOAD_MESH
    std::ifstream history("mesh.history");
#endif

    unsigned int ncycles=16-2*dim;

    for (unsigned int cycle=0; cycle<ncycles; ++cycle)
    {
      std::cout << "Cycle " << cycle << std::endl;

      if (cycle == 0)
      {
        create_tria(triangulation);
        if (dim == 2)
          triangulation.refine_global (5);
        else
          triangulation.refine_global (3);
      }
      else
      {

#ifdef LOAD_MESH
        triangulation.load_refine_flags (history);
        triangulation.load_coarsen_flags (history);
#else
        Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
        KellyErrorEstimator<dim>::estimate (dof_handler,
                                            QGauss<dim-1>(fe.degree+1),
                                            typename FunctionMap<dim>::type(),
                                            solution_host,
                                            estimated_error_per_cell);
        GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                         estimated_error_per_cell,
                                                         ((dim==3&&use_3d_l) ? 0.15 : 0.3), 0.03);

        triangulation.prepare_coarsening_and_refinement ();
#endif

#ifdef SAVE_MESH
        triangulation.save_refine_flags (history);
        triangulation.save_coarsen_flags (history);
#endif
        triangulation.execute_coarsening_and_refinement();
      }

#ifdef LAST_CYCLE
      if(cycle == ncycles-1)
#endif
      {


        // resets solution to 0:
        setup_system();

        interpolate_boundary_values();
        std::pair<unsigned int, std::pair<double,double> > stats = solve(false);
#ifndef NO_OUTPUT
        output_results(cycle);
#endif

        Vector<float> error_per_cell(triangulation.n_active_cells());
        VectorTools::integrate_difference (mapping,
                                           dof_handler,
                                           solution_host,
                                           Solution<dim>(),
                                           error_per_cell,
                                           QGauss<dim>(fe.degree+2),
                                           VectorTools::L2_norm);
        const double L2_error = sqrt(error_per_cell.norm_sqr());

        VectorTools::integrate_difference (mapping,
                                           dof_handler,
                                           solution_host,
                                           Solution<dim>(),
                                           error_per_cell,
                                           QGauss<dim>(fe.degree+1),
                                           VectorTools::H1_seminorm);
        const double grad_error = sqrt(error_per_cell.norm_sqr());

        convergence_table.add_value("cells", triangulation.n_global_active_cells());
        convergence_table.add_value("dofs",  dof_handler.n_dofs());
        convergence_table.add_value("val_L2",    L2_error);
        convergence_table.add_value("grad_L2",   grad_error);
        convergence_table.add_value("solver_its", stats.first);
        convergence_table.add_value("solver_time1", stats.second.first);
        convergence_table.add_value("solver_time2", stats.second.second);

        std::cout << std::endl;


        convergence_table.set_precision("val_L2", 3);
        convergence_table.set_scientific("val_L2", true);
        convergence_table.set_precision("grad_L2", 3);
        convergence_table.set_scientific("grad_L2", true);
        convergence_table.set_precision("solver_time1", 3);
        convergence_table.set_scientific("solver_time1", true);
        convergence_table.set_precision("solver_time2", 3);
        convergence_table.set_scientific("solver_time2", true);

        convergence_table.write_text(std::cout);
        convergence_table.write_tex(std::cout);
      }

    };
    // convergence_table.set_precision("val_L2", 3);
    // convergence_table.set_scientific("val_L2", true);
    // convergence_table.set_precision("grad_L2", 3);
    // convergence_table.set_scientific("grad_L2", true);
    // convergence_table.set_precision("solver_time1", 3);
    // convergence_table.set_scientific("solver_time1", true);
    // convergence_table.set_precision("solver_time2", 3);
    // convergence_table.set_scientific("solver_time2", true);

    // convergence_table.write_text(std::cout);
    // convergence_table.write_tex(std::cout);
  }
}


int main (int argc, char **argv)
{
  try
  {
    using namespace Step37;

    unsigned int degree = 3;
    if (argc > 1)
      degree = std::atoi(argv[1]);

    if (degree == 1)
      {
        LaplaceProblem<dimension, 1> laplace_problem;
        laplace_problem.run ();
      }
    else if (degree == 2)
      {
        LaplaceProblem<dimension, 2> laplace_problem;
        laplace_problem.run ();
      }
    else if (degree == 3)
      {
        LaplaceProblem<dimension, 3> laplace_problem;
        laplace_problem.run ();
      }
    else if (degree == 4)
      {
        LaplaceProblem<dimension, 4> laplace_problem;
        laplace_problem.run ();
      }
    else if (degree == 5)
      {
        LaplaceProblem<dimension, 5> laplace_problem;
        laplace_problem.run ();
      }
    else if (degree == 6)
      {
        LaplaceProblem<dimension, 6> laplace_problem;
        laplace_problem.run ();
      }
    else if (degree == 7)
      {
        LaplaceProblem<dimension, 7> laplace_problem;
        laplace_problem.run ();
      }
    else
      AssertThrow(false,
                  ExcNotImplemented("Degree " + std::to_string(degree) + " not implemented."));
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

  GrowingVectorMemory<GpuVector<Step37::level_number> >::release_unused_memory();
  GrowingVectorMemory<GpuVector<Step37::number> >::release_unused_memory();

  return 0;
}
