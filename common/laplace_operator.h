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
 * This file implements the LaplaceOperator with matrix-free operator
 * evaluation. The class supports both the constant-coefficient case and the
 * variable-coefficient case, specified by a function that is passed to the
 * function.
 */

#ifndef multigrid_laplace_operator_h
#define multigrid_laplace_operator_h

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>


namespace multigrid
{
  using namespace dealii;

  template <int dim, int fe_degree, typename number>
  class LaplaceOperator :
    public MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >
  {
  public:
    typedef number value_type;

    LaplaceOperator ();

    void
    initialize(std::shared_ptr<const MatrixFree<dim, number>> data,
               const std::vector<unsigned int> &mask);

    void
    initialize(std::shared_ptr<const MatrixFree<dim, number>> data,
               const MGConstrainedDoFs &        mg_constrained_dofs,
               const unsigned int               level);

    void clear();

    void vmult(LinearAlgebra::distributed::Vector<number> &dst,
               const LinearAlgebra::distributed::Vector<number> &src) const;

    void compute_residual (LinearAlgebra::distributed::Vector<number> &dst,
                           LinearAlgebra::distributed::Vector<number> &src,
                           const Function<dim>                        &rhs_function) const;

    void evaluate_coefficient(const Function<dim> &coefficient_function);

    virtual void compute_diagonal() override;

  private:
    virtual void apply_add(LinearAlgebra::distributed::Vector<number> &dst,
                           const LinearAlgebra::distributed::Vector<number> &src) const;

    void local_apply (const MatrixFree<dim,number>                     &data,
                      LinearAlgebra::distributed::Vector<number>       &dst,
                      const LinearAlgebra::distributed::Vector<number> &src,
                      const std::pair<unsigned int,unsigned int>       &cell_range) const;

    void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                                 LinearAlgebra::distributed::Vector<number>       &dst,
                                 const unsigned int                               &dummy,
                                 const std::pair<unsigned int,unsigned int>       &cell_range) const;

    void do_quadrature_point_operation(FEEvaluation<dim,fe_degree,fe_degree+1,1,number> &phi_in,
                                       FEEvaluation<dim,fe_degree,fe_degree+1,1,number> &phi_out,
                                       const unsigned int cell) const;

    void
    adjust_ghost_range_if_necessary(const LinearAlgebra::distributed::Vector<number> &vec) const;

    AlignedVector<Tensor<1,dim*(dim+1)/2,VectorizedArray<number>>> merged_coefficient;

    std::vector<unsigned int> vmult_edge_constrained_indices;

    mutable std::vector<double> vmult_edge_constrained_values;
  };



  template <int dim, int fe_degree, typename number>
  LaplaceOperator<dim,fe_degree,number>::LaplaceOperator ()
    :
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number> >()
  {
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::clear ()
  {
    MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::clear();
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::initialize (std::shared_ptr<const MatrixFree<dim, number>> data,
                const std::vector<unsigned int> &mask)
  {
    MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::initialize
      (data, mask);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::initialize (std::shared_ptr<const MatrixFree<dim, number>> data,
                const MGConstrainedDoFs &        mg_constrained_dofs,
                const unsigned int               level)
  {
    MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::initialize
      (data, mg_constrained_dofs, level, std::vector<unsigned int>({0}));

    std::vector<types::global_dof_index> interface_indices;
    mg_constrained_dofs
      .get_refinement_edge_indices(level)
      .fill_index_vector(interface_indices);
    vmult_edge_constrained_indices.clear();
    vmult_edge_constrained_indices.reserve(interface_indices.size());
    vmult_edge_constrained_values.resize(interface_indices.size());
    const IndexSet &locally_owned =
      this->data->get_dof_handler(0).locally_owned_mg_dofs(level);
    for (unsigned int i = 0; i < interface_indices.size(); ++i)
      if (locally_owned.is_element(interface_indices[i]))
        vmult_edge_constrained_indices.push_back(locally_owned.index_within_set(interface_indices[i]));
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::evaluate_coefficient (const Function<dim> &coefficient_function)
  {
    const bool is_constant_coefficient =
      dynamic_cast<const Functions::ConstantFunction<dim> *>(&coefficient_function) != nullptr;

    merged_coefficient.resize(is_constant_coefficient ?
                              this->data->get_mapping_info().cell_data[0].
                              jacobians[0].size()
                              :
                              this->data->n_cell_batches() *
                              this->data->get_n_q_points());
    FEEvaluation<dim,-1,-1,1,number> fe_eval(*this->data);

    for (unsigned int cell=0; cell<this->data->n_macro_cells(); ++cell)
      {
        const std::size_t data_ptr = this->data->get_mapping_info().
          cell_data[0].data_index_offsets[cell];
        if (is_constant_coefficient &&
            this->data->get_mapping_info().get_cell_type(cell)<2 )
          {
            Tensor<2,dim,VectorizedArray<number> > coef =
              this->data->get_mapping_info().cell_data[0].JxW_values[data_ptr] *
              transpose(this->data->get_mapping_info().cell_data[0].jacobians[0][data_ptr]) *
              this->data->get_mapping_info().cell_data[0].jacobians[0][data_ptr] *
              coefficient_function.value(Point<dim>());

            for (unsigned int d=0; d<dim; ++d)
              merged_coefficient[data_ptr][d] = coef[d][d];
            for (unsigned int c=0, d=0; d<dim; ++d)
              for (unsigned int e=d+1; e<dim; ++e, ++c)
                merged_coefficient[data_ptr][dim+c] = coef[d][e];
          }
        else
          {
            fe_eval.reinit(cell);
            for (unsigned int q=0; q<this->data->
                   get_mapping_info().cell_data[0].descriptor[0].quadrature_weights.size(); ++q)
              {
                VectorizedArray<number> function_value;
                Point<dim,VectorizedArray<number>> point_batch = fe_eval.quadrature_point(q);
                for (unsigned int v=0; v<VectorizedArray<number>::n_array_elements; ++v)
                  {
                    Point<dim> p;
                    for (unsigned int d=0; d<dim; ++d)
                      p[d] = point_batch[d][v];
                    function_value[v] = coefficient_function.value(p);
                  }

                // for affine geometries we also need to apply the quadrature
                // weight, for general geometries it is already in JxW_values
                if (this->data->get_mapping_info().get_cell_type(cell)<2)
                  function_value = function_value * this->data->get_mapping_info().
                    cell_data[0].descriptor[0].quadrature_weights[q];

                // for affine geometries there is only a single index, must
                // use them in all iterations
                const unsigned int stride = this->data->get_mapping_info().get_cell_type(cell) < 2 ? 0 : 1;
                Tensor<2,dim,VectorizedArray<number> > coef =
                  function_value *
                  this->data->get_mapping_info().cell_data[0].JxW_values[data_ptr+q*stride] *
                  transpose(this->data->get_mapping_info().cell_data[0].jacobians[0][data_ptr+q*stride]) *
                  this->data->get_mapping_info().cell_data[0].jacobians[0][data_ptr+q*stride];
                for (unsigned int d=0; d<dim; ++d)
                  merged_coefficient[cell*fe_eval.n_q_points+q][d] = coef[d][d];
                for (unsigned int c=0, d=0; d<dim; ++d)
                  for (unsigned int e=d+1; e<dim; ++e, ++c)
                    merged_coefficient[cell*fe_eval.n_q_points+q][dim+c] = coef[d][e];
              }
          }
      }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::do_quadrature_point_operation (FEEvaluation<dim,fe_degree,fe_degree+1,1,number> &phi_in,
                                   FEEvaluation<dim,fe_degree,fe_degree+1,1,number> &phi_out,
                                   const unsigned int cell) const
  {
    constexpr unsigned int n_q_points = Utilities::pow(fe_degree+1, dim);
    VectorizedArray<number> *phi_grads = phi_in.begin_gradients();
    VectorizedArray<number> *phi_grads_out = phi_out.begin_gradients();

    // affine geometry, constant coefficients
    if (this->data->get_mapping_info().get_cell_type(cell)<2 &&
        merged_coefficient.size() ==
        this->data->get_mapping_info().cell_data[0].jacobians[0].size())
      {
        const std::size_t data_ptr = this->data->get_mapping_info().
          cell_data[0].data_index_offsets[cell];
        for (unsigned int q=0; q<n_q_points; ++q)
          {
            const number weight = this->data->get_mapping_info().cell_data[0].descriptor[0].quadrature_weights[q];
            if (dim==2)
              {
                VectorizedArray<number> tmp = phi_grads[q];
                phi_grads_out[q] =
                  (merged_coefficient[data_ptr][0] * tmp +
                   merged_coefficient[data_ptr][2] * phi_grads[q+n_q_points]) * weight;
                phi_grads_out[q+n_q_points] =
                  (merged_coefficient[data_ptr][2] * tmp +
                   merged_coefficient[data_ptr][1] * phi_grads[q+n_q_points]) * weight;
              }

            else if (dim==3)
              {
                VectorizedArray<number> tmp0 = phi_grads[q];
                VectorizedArray<number> tmp1 = phi_grads[q+n_q_points];
                phi_grads_out[q] =
                  (merged_coefficient[data_ptr][0] * tmp0 +
                   merged_coefficient[data_ptr][3] * tmp1 +
                   merged_coefficient[data_ptr][4] * phi_grads[q+2*n_q_points]) * weight;
                phi_grads_out[q+n_q_points] =
                  (merged_coefficient[data_ptr][3] * tmp0 +
                   merged_coefficient[data_ptr][1] * tmp1 +
                   merged_coefficient[data_ptr][5] * phi_grads[q+2*n_q_points]) * weight;
                phi_grads_out[q+2*n_q_points] =
                  (merged_coefficient[data_ptr][4] * tmp0 +
                   merged_coefficient[data_ptr][5] * tmp1 +
                   merged_coefficient[data_ptr][2] * phi_grads[q+2*n_q_points]) * weight;
              }
            else
              AssertThrow(false, ExcMessage("Only dim=2,3 implemented"));
          }
      }
    // all other cases -> full array with coefficients
    else
      for (unsigned int q=0; q<n_q_points; ++q)
        {
          const std::size_t data_ptr = cell*n_q_points + q;
          if (dim==2)
            {
              VectorizedArray<number> tmp = phi_grads[q];
              phi_grads_out[q] =
                (merged_coefficient[data_ptr][0] * tmp +
                 merged_coefficient[data_ptr][2] * phi_grads[q+n_q_points]);
              phi_grads_out[q+n_q_points] =
                (merged_coefficient[data_ptr][2] * tmp +
                 merged_coefficient[data_ptr][1] * phi_grads[q+n_q_points]);
            }
          else if (dim==3)
            {
              VectorizedArray<number> tmp0 = phi_grads[q];
              VectorizedArray<number> tmp1 = phi_grads[q+n_q_points];
              phi_grads_out[q] =
                (merged_coefficient[data_ptr][0] * tmp0 +
                 merged_coefficient[data_ptr][3] * tmp1 +
                 merged_coefficient[data_ptr][4] * phi_grads[q+2*n_q_points]);
              phi_grads_out[q+n_q_points] =
                (merged_coefficient[data_ptr][3] * tmp0 +
                 merged_coefficient[data_ptr][1] * tmp1 +
                 merged_coefficient[data_ptr][5] * phi_grads[q+2*n_q_points]);
              phi_grads_out[q+2*n_q_points] =
                (merged_coefficient[data_ptr][4] * tmp0 +
                 merged_coefficient[data_ptr][5] * tmp1 +
                 merged_coefficient[data_ptr][2] * phi_grads[q+2*n_q_points]);
            }
          else
            AssertThrow(false, ExcMessage("Only dim=2,3 implemented"));
        }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::local_apply (const MatrixFree<dim,number>                     &data,
                 LinearAlgebra::distributed::Vector<number>       &dst,
                 const LinearAlgebra::distributed::Vector<number> &src,
                 const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        phi.gather_evaluate(src, false, true);
        do_quadrature_point_operation(phi, phi, cell);
        phi.integrate_scatter(false, true, dst);
      }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::apply_add (LinearAlgebra::distributed::Vector<number>       &dst,
               const LinearAlgebra::distributed::Vector<number> &src) const
  {
    this->data->cell_loop (&LaplaceOperator::local_apply, this, dst, src);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::vmult (LinearAlgebra::distributed::Vector<number>       &dst,
           const LinearAlgebra::distributed::Vector<number> &src) const
  {
    adjust_ghost_range_if_necessary(src);
    adjust_ghost_range_if_necessary(dst);

    for (unsigned int i = 0; i < vmult_edge_constrained_indices.size(); ++i)
      {
        vmult_edge_constrained_values[i] =
          src.local_element(vmult_edge_constrained_indices[i]);
        const_cast<LinearAlgebra::distributed::Vector<number> &>(src).
          local_element(vmult_edge_constrained_indices[i]) = 0.;
      }

    // zero dst within the loop
    this->data->cell_loop (&LaplaceOperator::local_apply, this, dst, src, true);

    for (auto i : this->data->get_constrained_dofs(0))
      dst.local_element(i) = src.local_element(i);
    for (unsigned int i = 0; i < vmult_edge_constrained_indices.size(); ++i)
      {
        dst.local_element(vmult_edge_constrained_indices[i]) =
          vmult_edge_constrained_values[i];
        const_cast<LinearAlgebra::distributed::Vector<number> &>(src).
          local_element(vmult_edge_constrained_indices[i]) = vmult_edge_constrained_values[i];
      }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::compute_diagonal ()
  {
    this->inverse_diagonal_entries.
    reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number> >());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
      this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    unsigned int dummy = 0;
    this->data->cell_loop (&LaplaceOperator::local_compute_diagonal, this,
                           inverse_diagonal, dummy);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
      {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
          1./inverse_diagonal.local_element(i);
      }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::local_compute_diagonal (const MatrixFree<dim,number>               &data,
                            LinearAlgebra::distributed::Vector<number> &dst,
                            const unsigned int &,
                            const std::pair<unsigned int,unsigned int> &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (data);

    AlignedVector<VectorizedArray<number> > diagonal(phi.dofs_per_cell);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        phi.reinit (cell);
        for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
              phi.submit_dof_value(VectorizedArray<number>(), j);
            phi.submit_dof_value(make_vectorized_array<number>(1.), i);
            phi.evaluate(false, true);
            do_quadrature_point_operation(phi, phi, cell);
            phi.integrate(false, true);

            diagonal[i] = phi.get_dof_value(i);
          }
        for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
          phi.submit_dof_value(diagonal[i], i);
        phi.distribute_local_to_global (dst);
      }
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::
  compute_residual (LinearAlgebra::distributed::Vector<number> &dst,
                    LinearAlgebra::distributed::Vector<number> &src,
                    const Function<dim>                        &rhs_function) const
  {
    dst = 0;
    src.update_ghost_values();
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi (*this->data, 0);
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> phi_nodirichlet (*this->data, 1);

    for (unsigned int cell=0; cell<this->data->n_cell_batches(); ++cell)
      {
        phi.reinit (cell);
        phi_nodirichlet.reinit (cell);
        phi_nodirichlet.read_dof_values(src);

        // change sign because we want to subtract the gradient term
        for (unsigned int i=0; i<phi_nodirichlet.dofs_per_cell; ++i)
          phi_nodirichlet.begin_dof_values()[i] *= make_vectorized_array<number>(-1.0);

        phi_nodirichlet.evaluate(false, true);
        do_quadrature_point_operation(phi_nodirichlet, phi, cell);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
          {
            Point<dim,VectorizedArray<number>> pvec = phi.quadrature_point(q);
            VectorizedArray<number> rhs_val;
            for (unsigned int v=0; v<VectorizedArray<number>::n_array_elements; ++v)
              {
                Point<dim> p;
                for (unsigned int d=0; d<dim; ++d)
                  p[d] = pvec[d][v];
                rhs_val[v] = rhs_function.value(p);
              }
            phi.submit_value(rhs_val, q);
          }
        phi.integrate_scatter(true, true, dst);
      }
    dst.compress(VectorOperation::add);
    src.zero_out_ghosts();
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>::
  adjust_ghost_range_if_necessary (const LinearAlgebra::distributed::Vector<number> &vec) const
  {
    if (vec.get_partitioner().get() ==
        this->data->get_dof_info(0).vector_partitioner.get())
      return;

    Assert(vec.get_partitioner()->local_size() ==
           this->data->get_dof_info(0).vector_partitioner->local_size(),
           ExcMessage("The vector passed to the vmult() function does not have "
                      "the correct size for compatibility with MatrixFree."));
    LinearAlgebra::distributed::Vector<number> copy_vec(vec);
    const_cast<LinearAlgebra::distributed::Vector<number> &>(vec)
          .reinit(this->data->get_dof_info(0).vector_partitioner);
    const_cast<LinearAlgebra::distributed::Vector<number> &>(vec)
      .copy_locally_owned_data_from(copy_vec);
  }

}

#endif
