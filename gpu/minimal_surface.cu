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
 * This program solves the minimal surface equation on a circle/ball, a
 * nonlinear variant of the Laplace equation. The nonlinearity is resolved
 * with Newton's method and a line search procedure for globalization. The
 * linear system is solved with the conjugate gradient method preconditioned
 * by a geometric multigrid V-cycle.
 */


#include <deal.II/base/subscriptor.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/matrix_free/operators.h>

#include <iostream>
#include <fstream>
#include <sstream>

#include "matrix_free_gpu/defs.h"
#include "matrix_free_gpu/gpu_vec.h"
#include "matrix_free_gpu/gpu_array.cuh"
#include "matrix_free_gpu/cuda_utils.cuh"
#include "matrix_free_gpu/mg_transfer_matrix_free_gpu.h"
#include "matrix_free_gpu/constraint_handler_gpu.h"
#include "matrix_free_gpu/matrix_free_gpu.h"
#include "matrix_free_gpu/fee_gpu.cuh"


// For accessign a tensor-valued coefficient with efficient storage

template <int dim, typename Number>
struct SymTensorView
{
  const static unsigned int n_values = dim*(dim+1)/2;
  __device__ static inline void assign(Number *array,
                                       const Number val,
                                       const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                                       const unsigned int cellid,
                                       const unsigned int q,
                                       const unsigned int i,
                                       const unsigned int j)
  {
    array[(gpu_data->rowstart+cellid*gpu_data->rowlength)*n_values + ((i*(i+1)/2)+j)*gpu_data->rowlength + q] = val;
  }

  __device__ static inline GpuArray<dim,Number> mul(const Number *array,
                                                    const GpuArray<dim,Number>& grad,
                                                    const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                                                    const unsigned int cellid,
                                                    const unsigned int q)
  {
    GpuArray<dim,Number> res;
#pragma unroll
    for(int i=0; i<dim; ++i) {
      Number t = 0;

#pragma unroll
      for(int j=0; j<=i; ++j)
        t += array[(gpu_data->rowstart+cellid*gpu_data->rowlength)*n_values + ((i*(i+1)/2)+j)*gpu_data->rowlength + q]*grad[j];

#pragma unroll
      for(int j=i+1; j<dim; ++j)
        t += array[(gpu_data->rowstart+cellid*gpu_data->rowlength)*n_values + ((j*(j+1)/2)+i)*gpu_data->rowlength + q]*grad[j];

      res[i] = t;
    }
    return res;
  }
};



namespace Step37
{

#ifdef DEGREE_FE
  const unsigned int degree_finite_element = DEGREE_FE;
#else
  const unsigned int degree_finite_element = 3;
#endif

#ifdef DIMENSION
  const unsigned int dimension = DIMENSION;
#else
  const unsigned int dimension = 2;
#endif

#ifdef USE_JACOBI
  const bool use_jacobi = true;
#else
  const bool use_jacobi = false;
#endif


  typedef double number;
  typedef float level_number;

  //---------------------------------------------------------------------------
  // operator
  //---------------------------------------------------------------------------

  template <int dim, int fe_degree, typename Number>
  class LaplaceOperator : public Subscriptor
  {
  public:
    typedef Number value_type;
    typedef GpuVector<Number> VectorType;

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
                           const VectorType &src, bool first_time=false) const;


    void evaluate_coefficient (const bool first_time,
                               const VectorType &solution);

    const std::shared_ptr<DiagonalMatrix<VectorType>> get_matrix_diagonal_inverse () const;


    void initialize_dof_vector(VectorType &v) const;

    std::size_t memory_consumption () const;


  private:
    unsigned int                level;

    std::shared_ptr<const MatrixFreeGpu<dim,Number>>    data;

    VectorType                  coefficient;

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
    const Number *coefficient;
    static const unsigned int n_dofs_1d = fe_degree+1;
    static const unsigned int n_q_points_1d = fe_degree+1;
    static const unsigned int n_local_dofs = ipow<n_dofs_1d,dim>::val;
    static const unsigned int n_q_points = ipow<n_q_points_1d,dim>::val;

    // what to do for each cell
    __device__ void cell_apply (Number                          *dst,
                                const Number                    *src,
                                const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                                const unsigned int cell,
                                SharedData<dim,Number> *shdata) const
    {
      FEEvaluationGpu<dim,fe_degree,Number> phi (cell, gpu_data, shdata);

      phi.read_dof_values(src);

      phi.evaluate (false,true);

      // apply the local operation
      const unsigned int q = (dim==1 ? threadIdx.x%n_q_points_1d :
                              dim==2 ? threadIdx.x%n_q_points_1d + n_q_points_1d*threadIdx.y :
                              threadIdx.x%n_q_points_1d + n_q_points_1d*(threadIdx.y + n_q_points_1d*threadIdx.z));

      phi.submit_gradient (SymTensorView<dim,Number>::mul(coefficient,phi.get_gradient(q),
                                                          gpu_data,cell,q), q);
      __syncthreads();

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
    constraint_handler.save_constrained_values(dst, const_cast<VectorType&>(src));

    // apply laplace operator
    LocalOperator<dim,fe_degree,Number> loc_op{coefficient.getDataRO()};
    data->cell_loop (dst,src,loc_op);

    // overwrite Dirichlet values in output with correct values, and reset input
    // to possibly non-zero values.
    constraint_handler.load_and_add_constrained_values(dst, const_cast<VectorType&>(src));
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
    LocalOperator<dim,fe_degree,Number> loc_op{coefficient.getDataRO()};
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
    LocalOperator<dim,fe_degree,Number> loc_op{coefficient.getDataRO()};
    data->cell_loop (dst,src_cpy,loc_op);

    // zero out edge values.
    // since boundary values should also be removed, do both at once.
    constraint_handler.set_constrained_values(dst,0.0);
  }


  template <int dim, int fe_degree, typename Number>
  struct ResidualLocalOperator {
    const bool first_time;
    static const unsigned int n_dofs_1d = fe_degree+1;
    static const unsigned int n_local_dofs = ipow<fe_degree+1,dim>::val;
    static const unsigned int n_q_points = ipow<fe_degree+1,dim>::val;

    // what to do on each quadrature point
    template <typename FEE>
    __device__ inline void quad_operation(FEE *phi, const unsigned int q) const
    {
      phi->submit_gradient (phi->get_gradient(q)*(first_time?(-1.0):(-1./std::sqrt(1.+phi->get_gradient(q).norm_square()))), q);
    }

    // what to do for each cell
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
  LaplaceOperator<dim,fe_degree,Number>::
  compute_residual (VectorType       &dst,
                    const VectorType &src, bool first_time) const
  {
    dst = 0;
    // leave constrained values as-is (we want the boundary values, and
    // hanging-node are never read anyways)

    ResidualLocalOperator<dim,fe_degree,Number> res_loc_op{first_time};
    data->cell_loop (dst,src,res_loc_op);

    // Set all constrained values to 0 (including hanging nodes, since we don't
    // use them later anyways)

    constraint_handler.set_constrained_values(dst,0);
  }


  template <int dim, int fe_degree, typename Number>
  struct DiagonalLocalOperator {
    const Number *coefficient;
    static const unsigned int n_dofs_1d = fe_degree+1;
    static const unsigned int n_q_points_1d = fe_degree+1;
    static const unsigned int n_local_dofs = ipow<n_dofs_1d,dim>::val;
    static const unsigned int n_q_points = ipow<n_q_points_1d,dim>::val;

    // what to do for each cell
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
        phi.submit_dof_value(i==tid?1.0:0.0,
                             tid);

        __syncthreads();
        phi.evaluate (false, true);

        phi.submit_gradient (SymTensorView<dim,Number>::mul(coefficient,phi.get_gradient(tid),
                                                            gpu_data,cell,tid),tid);

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

    DiagonalLocalOperator<dim,fe_degree,Number> diag_loc_op{coefficient.getDataRO()};
    data->cell_loop (inv_diag,diag_loc_op);

    constraint_handler.set_constrained_values(inv_diag,1.0);

    inv_diag.invert();

    diagonal_is_available = true;
  }




  template <int dim, int fe_degree, typename Number>
  struct CoefficientLocalOp
  {
    static const unsigned int n_dofs_1d = fe_degree+1;
    static const unsigned int n_q_points_1d = fe_degree+1;
    static const unsigned int n_local_dofs = ipow<n_dofs_1d,dim>::val;
    static const unsigned int n_q_points = ipow<n_q_points_1d,dim>::val;
    const bool first_time;

    // what to do for each cell
    __device__ void cell_apply (Number                                            *coefficient,
                                const Number                                      *solution,
                                const typename MatrixFreeGpu<dim,Number>::GpuData *gpu_data,
                                const unsigned int                                cell,
                                SharedData<dim,Number>                            *shdata) const
    {
      FEEvaluationGpu<dim,fe_degree,Number> phi (cell, gpu_data, shdata);

      const unsigned int q = (dim==1 ? threadIdx.x%n_q_points_1d :
                              dim==2 ? threadIdx.x%n_q_points_1d + n_q_points_1d*threadIdx.y :
                              threadIdx.x%n_q_points_1d + n_q_points_1d*(threadIdx.y + n_q_points_1d*threadIdx.z));

      phi.read_dof_values(solution);

      phi.evaluate (false,true);

      // apply local operation
      const GpuArray<dim,Number> grad = phi.get_gradient(q) *Number(first_time?0.0:1.0);
      const Number factor = 1./(1.+grad.norm_square());


      for(int i=0; i<dim; ++i)
        for(int j=0; j<=i; ++j)
          SymTensorView<dim,Number>::assign(coefficient,
                                            ((i==j) - factor*grad[i]*grad[j])*std::sqrt(factor),
                                            gpu_data,cell,q,i,j);

    }
  };


  template <int dim, int fe_degree, typename Number>
  void
  LaplaceOperator<dim,fe_degree,Number>::
  evaluate_coefficient (const bool first_time,
                        const GpuVector<Number> &solution)
  {

    coefficient.resize (data->n_cells_tot *
                        data->get_rowlength() *
                        SymTensorView<dim,Number>::n_values);

    CoefficientLocalOp<dim,fe_degree,Number> loc_op{first_time};
    data-> template cell_loop(coefficient,solution,loc_op);
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


  //---------------------------------------------------------------------------
  // solution
  //---------------------------------------------------------------------------

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution ()  : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };



  template <int dim>
  double Solution<dim>::value (const Point<dim>  &p,
                               const unsigned int) const
  {
    return std::sin(2*numbers::PI*(p[0] + p[1]));
  }





  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    void run ();

  private:
    void setup_system ();
    void solve (const bool first_time);
    void output_results (const unsigned int cycle) const;
    void interpolate_boundary_values();

    Triangulation<dim>                                               triangulation;

    MappingQGeneric<dim>                                             mapping;
    FE_Q<dim>                                                        fe;
    DoFHandler<dim>                                                  dof_handler;

    ConstraintMatrix                                                 constraints;
    std_cxx11::shared_ptr<MatrixFreeGpu<dim,number> >                system_matrix_free;
    typedef LaplaceOperator<dim,degree_finite_element,number>        SystemMatrixType;
    SystemMatrixType                                                 system_matrix;

    MGConstrainedDoFs                                                mg_constrained_dofs;
    MGTransferMatrixFreeGpu<dim,level_number>                        mg_transfer;
    typedef LaplaceOperator<dim,degree_finite_element,level_number>  LevelMatrixType;
    MGLevelObject<LevelMatrixType>                                   mg_matrices;

    GpuVector<number> solution;
    GpuVector<number> search_direction;
    GpuVector<number> tentative_solution;
    GpuVector<number> system_rhs;
    Vector<number>    solution_host;

    double                                     setup_time;
    ConditionalOStream                         time_details;

    double time_residual, time_solve;
    unsigned int n_residual, linear_iterations;
  };



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem ()
    :
    triangulation (Triangulation<dim>::limit_level_difference_at_vertices),
    mapping (degree_finite_element),
    fe (degree_finite_element),
    dof_handler (triangulation),
    mg_transfer (mg_constrained_dofs),
    time_details (std::cout,false),
    time_residual(0),
    time_solve(0),
    n_residual(0),
    linear_iterations(0)
  {}


  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    Timer time;
    time.start ();
    setup_time = 0;

    system_matrix_free.reset(new MatrixFreeGpu<dim,number>());
    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs (fe);
    dof_handler.distribute_mg_dofs (fe);

    std::cout << "Number of active cells:       "
              << triangulation.n_global_active_cells() << std::endl;

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
      system_matrix.initialize (system_matrix_free, constraints);
    }

    solution_host.reinit(system_matrix.m());
    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(search_direction);
    system_matrix.initialize_dof_vector(tentative_solution);
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
        std_cxx11::shared_ptr<MatrixFreeGpu<dim,level_number> >
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

    mg_transfer.build(dof_handler);

    setup_time += time.wall_time();
    time_details << "MG build transfer time     (CPU/wall) " << time()
                 << "s/" << time.wall_time() << "s\n";
    std::cout << "Total setup time               (wall) " << setup_time
          << "s\n";
  }



  template <int dim>
  void LaplaceProblem<dim>::interpolate_boundary_values()
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



  template <int dim>
  void LaplaceProblem<dim>::solve (const bool first_time)
  {
    Timer time;

    std::vector<GpuVector<level_number> > coefficient_solutions(triangulation.n_global_levels());
    coefficient_solutions.back() = solution;


   // do on CPU for now

    {
      for (unsigned int level=triangulation.n_global_levels()-1; level > 0; --level)
      {

        Vector<level_number> fine = coefficient_solutions[level].toVector();
        Vector<level_number> coarse(dof_handler.n_dofs(level-1));

        std::vector<float> dof_values_coarse(fe.dofs_per_cell);
        Vector<float> dof_values_fine(fe.dofs_per_cell);
        Vector<float> tmp(fe.dofs_per_cell);
        std::vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);


        typename DoFHandler<dim>::cell_iterator cell=dof_handler.begin(level-1);
        typename DoFHandler<dim>::cell_iterator endc=dof_handler.end(level-1);
        for ( ; cell != endc; ++cell)
          if (cell->is_locally_owned_on_level())
          {
            Assert(cell->has_children(), ExcNotImplemented());
            std::fill(dof_values_coarse.begin(), dof_values_coarse.end(), 0.);
            for (unsigned int child=0; child<cell->n_children(); ++child)
            {
              cell->child(child)->get_mg_dof_indices(dof_indices);
              for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
                dof_values_fine(i) = fine(dof_indices[i]);
              fe.get_restriction_matrix(child, cell->refinement_case()).vmult (tmp, dof_values_fine);
              for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
                if (fe.restriction_is_additive(i))
                  dof_values_coarse[i] += tmp[i];
                else if (tmp(i) != 0.)
                  dof_values_coarse[i] = tmp[i];
            }
            cell->get_mg_dof_indices(dof_indices);
            for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
              coarse(dof_indices[i]) = dof_values_coarse[i];
          }

        coefficient_solutions[level-1] = coarse;
      }

    }



    system_matrix.evaluate_coefficient(first_time, solution);


    typedef PreconditionChebyshev<LevelMatrixType, GpuVector<level_number> > SmootherType;
    mg::SmootherRelaxation<SmootherType, GpuVector<level_number> >
      mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
    {
      mg_matrices[level].evaluate_coefficient(first_time, coefficient_solutions[level]);
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

    ReductionControl solver_control (system_matrix.m(), 1e-13, 1e-4);
    SolverCG<GpuVector<number> > cg (solver_control);
    std::cout << "MG build smoother time     (CPU/wall) " << time()
          << "s/" << time.wall_time() << "s\n";

    time.reset();
    cudaDeviceSynchronize();
    time.start();

    system_matrix.compute_residual(system_rhs, solution, first_time);
    const double initial_residual_norm = system_rhs.l2_norm();
    cudaDeviceSynchronize();
    time.stop();
    time_residual += time.wall_time();
    ++n_residual;

    cudaDeviceSynchronize();
    time.restart();
    search_direction = 0;
    cg.solve (system_matrix, search_direction, system_rhs,
              preconditioner);

    cudaDeviceSynchronize();
    time.stop();
    time_solve += time.wall_time();
    linear_iterations += solver_control.last_step();

    std::cout << "Time solve ("
          << solver_control.last_step()
          << " iterations)  (CPU/wall) " << time() << "s/"
          << time.wall_time() << "s\n";

    // constraints.distribute(search_direction);

    cudaDeviceSynchronize();
    time.restart();
    double final_residual_norm = initial_residual_norm;
    double alpha = 1.;
    unsigned int n_steps = 0;
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

    cudaDeviceSynchronize();
    time.stop();
    time_residual += time.wall_time();

    std::cout << "Residual norm: " << initial_residual_norm << " in " << n_steps << " steps to " << final_residual_norm << std::endl;
    solution = tentative_solution;
    solution_host = solution.toVector();
  }




  template <int dim>
  void LaplaceProblem<dim>::output_results (const unsigned int cycle) const
  {
#ifndef NO_OUTPUT
    if (triangulation.n_global_active_cells() > 1000000)
      return;

    DataOut<dim> data_out;

    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution_host, "solution");
    data_out.build_patches (mapping, 1, DataOut<dim>::curved_inner_cells);

    std::ostringstream filename;
    filename << "solution-"
             << cycle
             << ".vtu";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);
#endif
  }




  template <int dim>
  void LaplaceProblem<dim>::run ()
  {
    const unsigned int n_inner_iterations = 100;
    for (unsigned int cycle=0; cycle<9-dim; ++cycle)
      {
        std::cout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
          {
            //GridGenerator::hyper_cube (triangulation, 0., 1.);
            GridGenerator::hyper_ball (triangulation);
            static const SphericalManifold<dim> boundary;
            triangulation.set_all_manifold_ids_on_boundary(0);
            triangulation.set_manifold (0, boundary);
            triangulation.refine_global (4-dim);
            setup_system();
          }
        else
          {
            triangulation.set_all_refine_flags();
            triangulation.prepare_coarsening_and_refinement ();
            triangulation.execute_coarsening_and_refinement();
            setup_system ();
          }

        interpolate_boundary_values();
        output_results (cycle*(1+n_inner_iterations) + 0);
        n_residual = 0;
        linear_iterations = 0;
        time_solve = 0;
        time_residual = 0;
        unsigned int inner_iteration = 0;
        for ( ; inner_iteration<n_inner_iterations; ++inner_iteration)
          {
            solve (inner_iteration==0);
            output_results (cycle*(1+n_inner_iterations) + inner_iteration + 1);
            if (system_rhs.l2_norm() < 1e-12)
              break;
          }
        std::cout << "Computing times: nl iterations: " << inner_iteration + 1 << " residuals "
                  << n_residual << " " << time_residual << "  linear solver " << linear_iterations << " " << time_solve << std::endl;
        std::cout << std::endl;
      };
  }
}




int main (int argc, char *argv[])
{
  try
    {
      using namespace Step37;

      LaplaceProblem<dimension> laplace_problem;
      laplace_problem.run ();
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
