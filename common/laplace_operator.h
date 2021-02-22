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

#include "../common/vector_access_reduced.h"

namespace multigrid
{
  using namespace dealii;

    void print_time(const double time,
                  const std::string &name,
                  const MPI_Comm communicator)
  {
    Utilities::MPI::MinMaxAvg data
      = Utilities::MPI::min_max_avg(time, communicator);

    if (Utilities::MPI::this_mpi_process(communicator)==0)
      {
        const unsigned int n_digits =
          std::ceil(std::log10(Utilities::MPI::n_mpi_processes(communicator)));
        std::cout << std::left << std::setw(29) << name << " "
                  << std::setw(11) << data.min
                  << " [p" << std::setw(n_digits) << data.min_index << "] "
                  << std::setw(11) << data.avg << " " << std::setw(11) << data.max
                  << " [p" << std::setw(n_digits) << data.max_index << "]"
                  << std::endl;
      }
  }



  template <int dim, int fe_degree, typename number>
  class LaplaceOperator :
    public MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >
  {
  public:
    typedef number value_type;

    LaplaceOperator ();

    void
    initialize(std::shared_ptr<const MatrixFree<dim, number>> data,
               const AffineConstraints<double> &constraints,
               const std::vector<unsigned int> &mask);

    void
    initialize(std::shared_ptr<const MatrixFree<dim, number>> data,
               const AffineConstraints<double> &constraints,
               const MGConstrainedDoFs &        mg_constrained_dofs,
               const unsigned int               level);

    void clear();

    void vmult(LinearAlgebra::distributed::Vector<number> &dst,
               const LinearAlgebra::distributed::Vector<number> &src) const;

    void vmult_residual
    (const LinearAlgebra::distributed::Vector<number> &rhs,
     const LinearAlgebra::distributed::Vector<number> &lhs,
     LinearAlgebra::distributed::Vector<number> &residual) const;

    std::array<number,4> vmult_with_cg_update
    (const number alpha,
     const number beta,
     const LinearAlgebra::distributed::Vector<number> &r,
     LinearAlgebra::distributed::Vector<number> &q,
     LinearAlgebra::distributed::Vector<number> &p,
     LinearAlgebra::distributed::Vector<number> &x) const;

    void vmult_with_chebyshev_update
    (const DiagonalMatrix<LinearAlgebra::distributed::Vector<number>> &prec,
     const LinearAlgebra::distributed::Vector<number> &rhs,
     const unsigned int iteration_index,
     const double factor1,
     const double factor2,
     LinearAlgebra::distributed::Vector<number> &solution,
     LinearAlgebra::distributed::Vector<number> &solution_old,
     LinearAlgebra::distributed::Vector<number> &temp_vector) const;

    void compute_residual (LinearAlgebra::distributed::Vector<number> &dst,
                           LinearAlgebra::distributed::Vector<number> &src,
                           const Function<dim>                        &rhs_function) const;

    void evaluate_coefficient(const Function<dim> &coefficient_function);

    virtual void compute_diagonal() override;

    unsigned int local_size_without_constraints() const
    {
      return first_constrained_index;
    }

    const std::vector<unsigned int>& get_compressed_dof_indices() const
    {
      return compressed_dof_indices;
    }

    const std::vector<unsigned char>& get_all_indices_uniform() const
    {
      return all_indices_uniform;
    }

  protected:
    AlignedVector<Tensor<1,dim*(dim+1)/2,VectorizedArray<number>>> merged_coefficient;

  private:
    void extract_compressed_indices(const AffineConstraints<double> &constraints,
                                    const unsigned int level);

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

    std::vector<unsigned int> vmult_edge_constrained_indices;

    mutable std::vector<double> vmult_edge_constrained_values;

    std::vector<unsigned int>  compressed_dof_indices;
    std::vector<unsigned char> all_indices_uniform;
    unsigned int               first_constrained_index;
  };



  template <int dim, int fe_degree, typename number>
  LaplaceOperator<dim,fe_degree,number>::LaplaceOperator ()
    :
    MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number> >(),
    first_constrained_index (0)
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
                const AffineConstraints<double> &constraints,
                const std::vector<unsigned int> &mask)
  {
    MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::initialize
      (data, mask);
    extract_compressed_indices(constraints, numbers::invalid_unsigned_int);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::initialize (std::shared_ptr<const MatrixFree<dim, number>> data,
                const AffineConstraints<double> &constraints,
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
    extract_compressed_indices(constraints, level);
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::extract_compressed_indices (const AffineConstraints<double> &constraints,
                                const unsigned int level)
  {
    if (fe_degree > 2)
      {
        compressed_dof_indices.resize(Utilities::pow(3,dim) *
                                      VectorizedArray<number>::size() *
                                      this->data->n_cell_batches(),
                                      numbers::invalid_unsigned_int);
        all_indices_uniform.resize(Utilities::pow(3,dim) *
                                   this->data->n_cell_batches(), 1);
      }
    std::vector<types::global_dof_index> dof_indices
      (this->data->get_dof_handler().get_fe().dofs_per_cell);
    for (unsigned int c=0; c<this->data->n_cell_batches(); ++c)
      {
        constexpr unsigned int n_lanes = VectorizedArray<number>::size();
        for (unsigned int l=0; l<this->data->n_active_entries_per_cell_batch(c); ++l)
          {
            if (fe_degree > 2)
              {
                if (level == dealii::numbers::invalid_unsigned_int)
                  this->data->get_cell_iterator(c, l)->get_dof_indices(dof_indices);
                else
                  typename dealii::DoFHandler<dim>::level_cell_iterator(this->data->get_cell_iterator(c, l))->get_active_or_mg_dof_indices(dof_indices);
                const unsigned int offset =
                  Utilities::pow(3,dim) * (n_lanes * c) + l;
                const Utilities::MPI::Partitioner &part =
                  *this->data->get_dof_info().vector_partitioner;
                unsigned int cc=0, cf=0;
                for (; cf<GeometryInfo<dim>::vertices_per_cell; ++cf, cc+=n_lanes)
                  if (!constraints.is_constrained(dof_indices[cf]))
                    compressed_dof_indices[offset+cc] = part.global_to_local(dof_indices[cf]);

                for (unsigned int line=0; line<GeometryInfo<dim>::lines_per_cell; ++line)
                  {
                    if (!constraints.is_constrained(dof_indices[cf]))
                      {
                        for (unsigned int i=0; i<fe_degree-1; ++i)
                          AssertThrow(dof_indices[cf+i] == dof_indices[cf]+i,
                                      ExcMessage("Expected contiguous numbering on level "
                                                 + std::to_string(level) + ", got c="
                                                 + std::to_string(c) + " l="
                                                 + std::to_string(l) + " i="
                                                 + std::to_string(i) + " df_idx[cf]="
                                                 + std::to_string(dof_indices[cf]) + " "
                                                 + std::to_string(dof_indices[cf+i]) + " l"
                                                 + std::to_string(line)));
                        compressed_dof_indices[offset+cc] = part.global_to_local(dof_indices[cf]);
                      }
                    cc += n_lanes;
                    cf += fe_degree-1;
                  }
                for (unsigned int quad=0; quad<GeometryInfo<dim>::quads_per_cell; ++quad)
                  {
                    if (!constraints.is_constrained(dof_indices[cf]))
                      {
                        for (unsigned int i=0; i<(fe_degree-1)*(fe_degree-1); ++i)
                          AssertThrow(dof_indices[cf+i] == dof_indices[cf]+i,
                                      ExcMessage("Expected contiguous numbering on level "
                                                 + std::to_string(level) + ", got c="
                                                 + std::to_string(c) + " l="
                                                 + std::to_string(l) + " i="
                                                 + std::to_string(i) + " df_idx[cf]="
                                                 + std::to_string(dof_indices[cf]) + " "
                                                 + std::to_string(dof_indices[cf+i]) + " q"
                                                 + std::to_string(quad)));
                        compressed_dof_indices[offset+cc] = part.global_to_local(dof_indices[cf]);
                      }
                    cc += n_lanes;
                    cf += (fe_degree-1)*(fe_degree-1);
                  }
                for (unsigned int hex=0; hex<GeometryInfo<dim>::hexes_per_cell; ++hex)
                  {
                    if (!constraints.is_constrained(dof_indices[cf]))
                      {
                        for (unsigned int i=0; i<(fe_degree-1)*(fe_degree-1)*(fe_degree-1); ++i)
                          AssertThrow(dof_indices[cf+i] == dof_indices[cf]+i,
                                      ExcMessage("Expected contiguous numbering on level "
                                                 + std::to_string(level) + ", got c="
                                                 + std::to_string(c) + " l="
                                                 + std::to_string(l) + " i="
                                                 + std::to_string(i) + " df_idx[cf]="
                                                 + std::to_string(dof_indices[cf]) + " "
                                                 + std::to_string(dof_indices[cf+i]) + " h"
                                                 + std::to_string(hex)));
                        compressed_dof_indices[offset+cc] = part.global_to_local(dof_indices[cf]);
                      }
                    cc += n_lanes;
                    cf += (fe_degree-1)*(fe_degree-1)*(fe_degree-1);
                  }
                AssertThrow(cc == n_lanes*Utilities::pow(3,dim),
                            ExcMessage("Expected 3^dim dofs, got " + std::to_string(cc)));
                AssertThrow(cf == dof_indices.size(),
                            ExcMessage("Expected (fe_degree+1)^dim dofs, got " +
                                       std::to_string(cf)));
              }
          }
        if (fe_degree > 2)
          {
            for (unsigned int i=0; i<Utilities::pow<unsigned int>(3,dim); ++i)
              for (unsigned int v=0; v<n_lanes; ++v)
                if (compressed_dof_indices[Utilities::pow(3,dim) * (n_lanes * c) + i*n_lanes + v] == numbers::invalid_unsigned_int)
                  all_indices_uniform[Utilities::pow(3,dim) * c + i] = 0;
          }
      }

    for (unsigned int i=0; i<this->data->get_constrained_dofs(0).size(); ++i)
      AssertThrow(this->data->get_constrained_dofs(0)[i] ==
                  this->data->get_constrained_dofs(0)[0]+i,
                  ExcMessage("Expected contiguous constrained dofs, got "
                             + std::to_string(this->data->get_constrained_dofs(0)[i])
                             + " vs "
                             + std::to_string(this->data->get_constrained_dofs(0)[0])
                             + " + " + std::to_string(i)));

    if (!this->data->get_constrained_dofs(0).empty())
      AssertThrow(this->data->get_constrained_dofs(0).back() ==
                  this->data->get_dof_info(0).vector_partitioner->local_size()-1,
                  ExcMessage("Expected constrained dofs at the end of locally owned dofs"));

    // set first constrained dof to skip operations in the Chebyshev smoother
    // for those entries (but not on level 0)
    first_constrained_index = this->data->get_constrained_dofs(0).empty() ||
      this->data->get_mg_level()==0 ?
      this->data->get_dof_info(0).vector_partitioner->local_size() :
      this->data->get_constrained_dofs(0)[0];
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
    FEEvaluation<dim,fe_degree,fe_degree+1,1,number> fe_eval(*this->data);

    for (unsigned int cell=0; cell<this->data->n_cell_batches(); ++cell)
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
                for (unsigned int v=0; v<VectorizedArray<number>::size(); ++v)
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
        if (fe_degree > 2)
          {
            read_dof_values_compressed<dim,fe_degree,number>
              (src, compressed_dof_indices, all_indices_uniform, cell, phi.begin_dof_values());
            phi.evaluate(false, true);
          }
        else
          phi.gather_evaluate(src, false, true);
        do_quadrature_point_operation(phi, phi, cell);
        if (fe_degree > 2)
          {
            phi.integrate(false, true);
            distribute_local_to_global_compressed<dim,fe_degree,number>
              (dst, compressed_dof_indices, all_indices_uniform, cell, phi.begin_dof_values());
          }
        else
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

    for (unsigned int i=0; i<this->data->get_constrained_dofs(0).size(); ++i)
      dst.local_element(i+this->data->get_constrained_dofs(0)[0]) =
        src.local_element(i+this->data->get_constrained_dofs(0)[0]);
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
  ::vmult_residual
    (const LinearAlgebra::distributed::Vector<number> &rhs,
     const LinearAlgebra::distributed::Vector<number> &lhs,
     LinearAlgebra::distributed::Vector<number> &residual) const
  {
    this->data->
      cell_loop(&LaplaceOperator::local_apply,
                this, residual, lhs,
                [&](const unsigned int start_range,
                    const unsigned int end_range)
                {
                  // zero 'temp_vector' before local_apply()
                  if (end_range > start_range)
                    std::memset(residual.begin()+start_range,
                                0,
                                sizeof(number)*(end_range-start_range));
                },
                [&](const unsigned int start_range,
                    const unsigned int end_range)
                {
                  if (!this->data->get_constrained_dofs(0).empty() &&
                      end_range > this->data->get_constrained_dofs(0)[0])
                    {
                      DEAL_II_OPENMP_SIMD_PRAGMA
                      for (unsigned int i=std::max(start_range,
                                                   this->data->get_constrained_dofs(0)[0]);
                           i<end_range; ++i)
                        residual.local_element(i) = lhs.local_element(i);
                    }

                  const number* rhs_ptr = rhs.begin();
                  number* residual_ptr = residual.begin();

                  DEAL_II_OPENMP_SIMD_PRAGMA
                  for (unsigned int i=start_range; i<end_range; ++i)
                    residual_ptr[i] = rhs_ptr[i] - residual_ptr[i];
                });
  }



  template <int dim, int fe_degree, typename number>
  std::array<number,4>
  LaplaceOperator<dim,fe_degree,number>
  ::vmult_with_cg_update (const number alpha,
                          const number beta,
                          const LinearAlgebra::distributed::Vector<number> &r,
                          LinearAlgebra::distributed::Vector<number> &q,
                          LinearAlgebra::distributed::Vector<number> &p,
                          LinearAlgebra::distributed::Vector<number> &x) const
  {
    using Simd = VectorizedArray<number>;
    std::array<Simd,4> sums = {};
    this->data->
      cell_loop(&LaplaceOperator::local_apply,
                this, q, p,
                [&](const unsigned int start_range,
                    const unsigned int end_range)
                {
                  // update x vector with old content of p
                  // update p vector according to beta formula
                  // zero q vector (after having read p)
                  Simd *arr_q = reinterpret_cast<Simd*>(q.begin()+start_range);
                  Simd *arr_p = reinterpret_cast<Simd*>(p.begin()+start_range);
                  Simd *arr_x = reinterpret_cast<Simd*>(x.begin()+start_range);
                  if (alpha == number())
                    {
                      for (unsigned int i=0; i<(end_range - start_range)/Simd::size(); ++i)
                        {
                          arr_p[i] = arr_q[i];
                          arr_q[i] = Simd();
                        }
                      for (unsigned int i=end_range/Simd::size()*Simd::size(); i<end_range; ++i)
                        {
                          p.local_element(i) = q.local_element(i);
                          q.local_element(i) = 0.;
                        }
                    }
                  else
                    {
                      for (unsigned int i=0; i<(end_range - start_range)/Simd::size(); ++i)
                        {
                          arr_x[i] += alpha * arr_p[i];
                          arr_p[i] = beta * arr_p[i] + arr_q[i];
                          arr_q[i] = Simd();
                        }
                      for (unsigned int i=end_range/Simd::size()*Simd::size(); i<end_range; ++i)
                        {
                          x.local_element(i) += alpha * p.local_element(i);
                          p.local_element(i) = beta * p.local_element(i) + q.local_element(i);
                          q.local_element(i) = 0.;
                        }
                    }
                },
                [&](const unsigned int start_range,
                    const unsigned int end_range)
                {
                  if (!this->data->get_constrained_dofs(0).empty() &&
                      end_range > this->data->get_constrained_dofs(0)[0])
                    for (unsigned int i=std::max(start_range,
                                                 this->data->get_constrained_dofs(0)[0]);
                         i<end_range; ++i)
                      q.local_element(i) = p.local_element(i);

                  const Simd *arr_q = reinterpret_cast<const Simd*>(q.begin()+start_range);
                  const Simd *arr_p = reinterpret_cast<const Simd*>(p.begin()+start_range);
                  const Simd *arr_r = reinterpret_cast<const Simd*>(r.begin()+start_range);

                  for (unsigned int i=0; i<(end_range - start_range)/Simd::size(); ++i)
                    {
                      sums[0] += arr_q[i] * arr_p[i];
                      sums[1] += arr_r[i] * arr_r[i];
                      sums[2] += arr_q[i] * arr_r[i];
                      sums[3] += arr_q[i] * arr_q[i];
                    }
                  for (unsigned int i=end_range/Simd::size()*Simd::size(); i<end_range; ++i)
                    {
                      sums[0][0] += q.local_element(i) * p.local_element(i);
                      sums[1][0] += r.local_element(i) * r.local_element(i);
                      sums[2][0] += q.local_element(i) * r.local_element(i);
                      sums[3][0] += q.local_element(i) * q.local_element(i);
                    }
                });
    std::array<number,4> results = {};
    for (unsigned int i=0; i<4; ++i)
      for (unsigned int v=0; v<Simd::size(); ++v)
        results[i] += sums[i][v];
    Utilities::MPI::sum(ArrayView<const number>(results.data(), 4),
                        q.get_mpi_communicator(),
                        ArrayView<number>(results.data(), 4));
    return results;
  }



  template <int dim, int fe_degree, typename number>
  void
  LaplaceOperator<dim,fe_degree,number>
  ::vmult_with_chebyshev_update
    (const DiagonalMatrix<LinearAlgebra::distributed::Vector<number>> &prec,
     const LinearAlgebra::distributed::Vector<number> &rhs,
     const unsigned int iteration_index,
     const double factor1,
     const double factor2,
     LinearAlgebra::distributed::Vector<number> &solution,
     LinearAlgebra::distributed::Vector<number> &solution_old,
     LinearAlgebra::distributed::Vector<number> &temp_vector) const
  {
    //#ifdef LIKWID_PERFMON
    //LIKWID_MARKER_START(("vmult_cheby_" + std::to_string(this->data->get_mg_level())).c_str());
    //#endif
    if (iteration_index > 0)
      {
        this->data->
          cell_loop(&LaplaceOperator::local_apply,
                    this, temp_vector, solution,
                    [&](const unsigned int start_range,
                        const unsigned int end_range)
                    {
                      // zero 'temp_vector' before local_apply()
                      const unsigned int my_end_range = std::min(first_constrained_index, end_range);
                      if (my_end_range > start_range)
                        std::memset(temp_vector.begin()+start_range,
                                    0,
                                    sizeof(number)*(my_end_range-start_range));
                    },
                    [&](const unsigned int start_range,
                        const unsigned int end_range)
                    {
                      if (this->data->get_mg_level() == 0 &&
                          !this->data->get_constrained_dofs(0).empty() &&
                          end_range > this->data->get_constrained_dofs(0)[0])
                        for (unsigned int i=std::max(start_range,
                                                     this->data->get_constrained_dofs(0)[0]);
                             i<end_range; ++i)
                          temp_vector.local_element(i) = solution.local_element(i);

                      // run the vector updates of Chebyshev after
                      // local_apply()
                      const unsigned int my_end_range = std::min(first_constrained_index, end_range);
                      if (my_end_range > start_range)
                        {
                          internal::PreconditionChebyshevImplementation
                            ::VectorUpdater<number>
                            updater(rhs.begin(), prec.get_vector().begin(),
                                    iteration_index, factor1, factor2,
                                    solution_old.begin(), temp_vector.begin(),
                                    solution.begin());
                          updater.apply_to_subrange(start_range, my_end_range);
                        }
                    });
        if (iteration_index == 1)
          {
            solution.swap(temp_vector);
            solution_old.swap(temp_vector);
          }
        else
          solution.swap(solution_old);
      }
    else
      {
        internal::PreconditionChebyshevImplementation::VectorUpdater<number>
          updater(rhs.begin(), prec.get_vector().begin(), iteration_index,
                  factor1, factor2, solution_old.begin(), temp_vector.begin(),
                  solution.begin());
        updater.apply_to_subrange(0U, rhs.local_size());
      }
    //#ifdef LIKWID_PERFMON
    //LIKWID_MARKER_STOP(("vmult_cheby_" + std::to_string(this->data->get_mg_level())).c_str());
    //#endif
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
            for (unsigned int v=0; v<VectorizedArray<number>::size(); ++v)
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
