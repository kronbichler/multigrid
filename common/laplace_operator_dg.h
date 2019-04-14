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

#ifndef multigrid_laplace_operator_dg_h
#define multigrid_laplace_operator_dg_h

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#define ALWAYS_INLINE DEAL_II_ALWAYS_INLINE
#include "matrix_vector_kernel.h"
#include "vector_access_reduced.h"
#include "laplace_operator.h"

//#define DO_GL_INSTEAD_OF_FDM 1
const double penalty_factor = 3.;

namespace multigrid
{
  using namespace dealii;

  template <typename Number>
  void read_dg (const unsigned int n_filled_components,
                const unsigned int size,
                const Number *src,
                const unsigned int *indices,
                VectorizedArray<Number> *dst)
  {
    if (n_filled_components == VectorizedArray<Number>::n_array_elements)
      vectorized_load_and_transpose(size, src, indices, dst);
    else
      {
        for (unsigned int i=0; i<size; ++i)
          dst[i] = {};
        for (unsigned int v=0;  v<n_filled_components; ++v)
          for (unsigned int i=0; i<size; ++i)
            dst[i][v] = src[indices[v] + i];
      }
  }

  template <typename Number>
  void write_dg (const unsigned int n_filled_components,
                 const bool add_into,
                 const unsigned int size,
                 VectorizedArray<Number> *src,
                 const unsigned int *indices,
                 Number *dst)
  {
    if (n_filled_components == VectorizedArray<Number>::n_array_elements)
      vectorized_transpose_and_store(add_into, size, src, indices, dst);
    else
      {
        for (unsigned int v=0;  v<n_filled_components; ++v)
          for (unsigned int i=0; i<size; ++i)
            dst[indices[v] + i] = src[i][v];
      }
  }



  template <int dim, int type, int degree, typename Number>
  class LocalBasisTransformer
  {
  public:
    LocalBasisTransformer(const MatrixFree<dim,Number> &mf,
                          const unsigned int dof_index)
      :
      transformation_is_hierarchical(false)
    {
      std::string fen = mf.get_dof_handler(dof_index).get_fe().get_name();
      fen[fen.find_first_of('<')+1] = '1';
      std::unique_ptr<FiniteElement<1> > fe(FETools::get_fe_by_name<1,1>(fen));
      const FiniteElement<1> &fe_1d = *fe;
      const unsigned int N = fe_1d.dofs_per_cell;
      if (type == 1)
        {
          QGaussLobatto<1> quad(N);
          const unsigned int stride = (N+1)/2;
          transformation_matrix.resize(N*stride);
          transformation_matrix_inverse.resize(N*stride);
          FullMatrix<double> shapes(N,N);
          for (unsigned int i=0; i<N; ++i)
            for (unsigned int j=0; j<N; ++j)
              shapes(i,j) = fe_1d.shape_value(i, quad.point(j));

          //shapes.print_formatted(std::cout);

          for (unsigned int i=0; i<N/2; ++i)
            for (unsigned int q=0; q<stride; ++q)
              {
                transformation_matrix[i*stride+q] = 0.5* (shapes(i,q) + shapes(i, N-1-q));
                transformation_matrix[(N-1-i)*stride+q] = 0.5* (shapes(i,q) - shapes(i, N-1-q));
              }
          if (N % 2 == 1)
            for (unsigned int q=0; q<stride; ++q)
              transformation_matrix[N/2*stride+q] = shapes(N/2,q);
          shapes.gauss_jordan();
          for (unsigned int i=0; i<N/2; ++i)
            for (unsigned int q=0; q<stride; ++q)
              {
                transformation_matrix_inverse[i*stride+q] = 0.5* (shapes(i,q) + shapes(i, N-1-q));
                transformation_matrix_inverse[(N-1-i)*stride+q] = 0.5* (shapes(i,q) - shapes(i, N-1-q));
              }
          if (N % 2 == 1)
            for (unsigned int q=0; q<stride; ++q)
              transformation_matrix_inverse[N/2*stride+q] = shapes(N/2,q);
        }
      else
        {
          LAPACKFullMatrix<double> mass_1d(N, N);
          LAPACKFullMatrix<double> lapl_1d(N, N);
          {
            QGauss<1> quad(N);
            for (unsigned int i=0; i<N; ++i)
              for (unsigned int j=0; j<N; ++j)
                {
                  double sum_m = 0, sum_l = 0;
                  for (unsigned int q=0; q<quad.size(); ++q)
                    {
                      sum_m += (fe_1d.shape_value(i, quad.point(q)) *
                                fe_1d.shape_value(j, quad.point(q))
                                )* quad.weight(q);
                      sum_l += (fe_1d.shape_grad(i, quad.point(q))[0] *
                               fe_1d.shape_grad(j, quad.point(q))[0]
                          )* quad.weight(q);
                    }
                  mass_1d(i,j) = sum_m;
                  const double penalty = (N)*(N)*penalty_factor;
                  sum_l += (fe_1d.shape_value(i, Point<1>()) *
                            fe_1d.shape_value(j, Point<1>()) * penalty
                            +
                            0.5*fe_1d.shape_grad(i, Point<1>())[0] *
                           fe_1d.shape_value(j, Point<1>())
                           +
                           0.5*fe_1d.shape_grad(j, Point<1>())[0] *
                      fe_1d.shape_value(i, Point<1>()));
                  sum_l += (1.*fe_1d.shape_value(i, Point<1>(1.0)) *
                            fe_1d.shape_value(j, Point<1>(1.0)) * penalty
                            -
                            0.5*fe_1d.shape_grad(i, Point<1>(1.0))[0] *
                           fe_1d.shape_value(j, Point<1>(1.0))
                           -
                           0.5*fe_1d.shape_grad(j, Point<1>(1.0))[0] *
                      fe_1d.shape_value(i, Point<1>(1.0)));
                  lapl_1d(i,j) = sum_l;
                }
          }

          std::vector<dealii::Vector<double> > eigenvecs(N);
          lapl_1d.compute_generalized_eigenvalues_symmetric(mass_1d, eigenvecs);

          transformation_matrix.resize(N*N);
          for (unsigned int i=0; i<N; ++i)
            for (unsigned int j=0; j<N; ++j)
              transformation_matrix[i*N+j] = eigenvecs[i][j];

          //for (unsigned int i=0; i<N; ++i)
          //  std::cout << lapl_1d.eigenvalue(i).real() << " ";
          //std::cout << std::endl;

          // check if the transformation matrix is symmetric
          transformation_is_hierarchical = true;
          for (unsigned int i=0; i<N; ++i)
            {
              for (unsigned int j=0; j<N/2; ++j)
                {
                  if (i%2 == 1 &&
                      std::abs(transformation_matrix[i*N+j][0]+
                               transformation_matrix[(i+1)*N-1-j][0]) >
                      1000.*std::numeric_limits<Number>::epsilon())
                    transformation_is_hierarchical = false;
                  if (i%2 == 0 &&
                      std::abs(transformation_matrix[i*N+j][0]-
                               transformation_matrix[(i+1)*N-1-j][0]) >
                      1000.*std::numeric_limits<Number>::epsilon())
                    transformation_is_hierarchical = false;
                }
            }
        }
      //std::cout << "Transformation is symmetric: " << transformation_is_symmetric << std::endl;

      /*
      std::cout << "Laplace:" << std::endl;
      lapl_1d.print_formatted(std::cout, 7);
      std::cout << "Mass:" << std::endl;
      mass_1d.print_formatted(std::cout, 7);

      std::cout << "Eigenvalues generic:" << std::endl;
      LAPACKFullMatrix<Number> lapl_copy(lapl_1d);
      LAPACKFullMatrix<Number> mass_copy(mass_1d);
      std::vector<dealii::Vector<Number> > eigenvecs(lapl_1d.m());
      lapl_copy.compute_generalized_eigenvalues_symmetric(mass_copy, eigenvecs);
      for (unsigned int i=0; i<lapl_copy.m(); ++i)
        std::cout << lapl_copy.eigenvalue(i).real() << " ";
      std::cout << std::endl;
      std::cout << "Eigenvectors generic:" << std::endl;
      for (unsigned int i=0; i<lapl_copy.m(); ++i)
        {
          for (unsigned int j=0; j<lapl_copy.m(); ++j)
            std::cout << eigenvecs[i][j] << " ";
          std::cout << std::endl;
        }
      std::cout << "Eigenvalues scaled:" << std::endl;
      lapl_copy = lapl_1d;
      lapl_copy *= 16.;
      mass_copy = mass_1d;
      mass_copy *= 1./16.;
      lapl_copy.compute_generalized_eigenvalues_symmetric(mass_copy, eigenvecs);
      for (unsigned int i=0; i<lapl_copy.m(); ++i)
        std::cout << lapl_copy.eigenvalue(i).real() << " ";
      std::cout << std::endl;
      std::cout << "Eigenvectors scaled:" << std::endl;
      for (unsigned int i=0; i<lapl_copy.m(); ++i)
        {
          for (unsigned int j=0; j<lapl_copy.m(); ++j)
            std::cout << eigenvecs[i][j] << " ";
          std::cout << std::endl;
        }
      */
    }

    template <bool transpose>
    void apply(const VectorizedArray<Number> *input,
               VectorizedArray<Number> *output) const
    {
      if (transformation_is_hierarchical)
        {
          internal::EvaluatorTensorProduct<internal::evaluate_symmetric_hierarchical,dim,
                                           degree+1,degree+1,VectorizedArray<Number> >
              eval(transformation_matrix, transformation_matrix, transformation_matrix);
          eval.template values<0,!transpose,false>(input,output);
          if (dim>1)
            eval.template values<1,!transpose,false>(output,output);
          if (dim>2)
            eval.template values<2,!transpose,false>(output,output);
        }
      else
        {
          internal::EvaluatorTensorProduct<internal::evaluate_evenodd,dim,
                                           degree+1,degree+1,VectorizedArray<Number> >
              eval(transformation_matrix_inverse);
          eval.template values<0,!transpose,false>(input,output);
          if (dim>1)
            eval.template values<1,!transpose,false>(output,output);
          if (dim>2)
            eval.template values<2,!transpose,false>(output,output);
        }
    }

  private:

    AlignedVector<VectorizedArray<Number> > transformation_matrix;
    AlignedVector<VectorizedArray<Number> > transformation_matrix_inverse;
    bool transformation_is_hierarchical;
  };



  template <int, int, typename> class JacobiTransformed;



  template <int dim, int fe_degree, typename Number>
  class LaplaceOperatorCompactCombine : public Subscriptor
  {
  public:
    typedef Number value_type;

    LaplaceOperatorCompactCombine() {}

    void reinit (std::shared_ptr<const MatrixFree<dim, Number>> data,
                 const unsigned int               dg_dof_index,
                 const LaplaceOperator<dim,fe_degree,Number> *op_fe_in = nullptr)
    {
      this->op_fe = op_fe_in;
      this->matrixfree = data;
      dof_index_dg = dg_dof_index;

      local_basis_transformer.reset(new LocalBasisTransformer<dim,1,fe_degree,Number>(*data, dg_dof_index));

      Assert(fe_degree >= 3, ExcNotImplemented());

      {
        FE_DGQArbitraryNodes<1> fe_collocation (QGauss<1>(fe_degree+1));
        shape_values_on_face_eo.resize(2*(fe_degree+1));
        for (unsigned int i=0; i<fe_degree/2+1; ++i)
          {
            const double v0 = fe_collocation.shape_value(i, Point<1>(0.));
            const double v1 = fe_collocation.shape_value(i, Point<1>(1.));
            shape_values_on_face_eo[fe_degree-i] = 0.5 * (v0 - v1);
            shape_values_on_face_eo[i] = 0.5 * (v0 + v1);

            const double d0 = fe_collocation.shape_grad(i, Point<1>(0.))[0];
            const double d1 = fe_collocation.shape_grad(i, Point<1>(1.))[0];
            shape_values_on_face_eo[fe_degree+1+i] = 0.5 * (d0 + d1);
            shape_values_on_face_eo[fe_degree+1+fe_degree-i] = 0.5 * (d0 - d1);
          }
      }
      hermite_derivative_on_face = matrixfree->get_shape_info(dof_index_dg).shape_data_on_face[0][fe_degree+1];
      FEFaceValues<dim> fe_face_values(matrixfree->get_dof_handler().get_fe(),
                                       QGauss<dim-1>(1),
                                       update_normal_vectors |
                                       update_jacobians |
                                       update_JxW_values);
      if (matrixfree->n_macro_cells() > 0)
        for (unsigned int d=0; d<dim; ++d)
          {
            unsigned int face_derivative_order_3d[3][3] = {{1, 2, 0}, {2, 0, 1}, {0, 1, 2}};
            unsigned int face_derivative_order_2d[2][2] = {{1, 0}, {0, 1}};
            unsigned int * face_der = dim==3?
                                      &face_derivative_order_3d[0][0] :
                                      &face_derivative_order_2d[0][0];
            fe_face_values.reinit(matrixfree->get_cell_iterator(0,0), 2*d);
            DerivativeForm<1, dim, dim> inv_jac =
              fe_face_values.jacobian(0).covariant_form();
            const Tensor<1,dim> normal = fe_face_values.normal_vector(0);
            const Tensor<1,dim> normal_times_jacobian =
                normal * Tensor<2,dim>(inv_jac);
            for (unsigned int e=0; e<dim; ++e)
              normal_jac1[d][e] = normal_times_jacobian[face_der[d*dim+e]];
            for (unsigned int e=0; e<dim; ++e)
              normal_jac2[d][e] = -normal_times_jacobian[face_der[d*dim+e]];
            for (unsigned int e=0; e<dim; ++e)
            normal_vector[d][e] = -normal[e];
            face_jxw[d] = fe_face_values.JxW(0);
            my_sigma[d] = std::abs(normal_times_jacobian[face_der[d*dim+dim-1]]);
          }
      {
        face_quadrature_weights.resize(Utilities::pow(fe_degree+1,dim-1));
        QGauss<dim-1> gauss(fe_degree+1);
        for (unsigned int i=0; i<face_quadrature_weights.size(); ++i)
          face_quadrature_weights[i] = gauss.weight(i);
      }

      std::map<std::pair<int,unsigned int>, unsigned int> map_to_mf_numbering;
      for (unsigned int cell=0; cell<matrixfree->n_macro_cells(); ++cell)
        for (unsigned int v=0; v<matrixfree->n_components_filled(cell); ++v)
          {
            const typename DoFHandler<dim>::cell_iterator dcell=matrixfree->get_cell_iterator(cell, v);
            map_to_mf_numbering[std::make_pair(dcell->level(),dcell->index())]
              = cell*VectorizedArray<Number>::n_array_elements+v;
          }
      start_indices_on_neighbor.reinit(TableIndices<3>(matrixfree->n_macro_cells(),
                                                       GeometryInfo<dim>::faces_per_cell,
                                                       VectorizedArray<Number>::n_array_elements));

      all_owned_faces.reinit(TableIndices<2>(matrixfree->n_macro_cells(), 2*dim));
      all_owned_faces.fill(static_cast<unsigned char>(1));
      dirichlet_faces.reinit(TableIndices<3>(matrixfree->n_macro_cells(), 2*dim,
                                             VectorizedArray<Number>::n_array_elements));
      start_indices_auxiliary.reinit(start_indices_on_neighbor.size());
      start_indices_auxiliary.fill(numbers::invalid_unsigned_int);

      std::map<unsigned int, std::vector<std::array<types::global_dof_index,5> > > proc_neighbors;
      std::vector<types::global_dof_index> dof_indices(matrixfree->get_dof_handler(dof_index_dg).get_fe().dofs_per_cell);
      for (unsigned int cell=0; cell<matrixfree->n_macro_cells(); ++cell)
        for (unsigned int v=0; v<matrixfree->n_components_filled(cell); ++v)
          {
            Assert(matrixfree->get_dof_info(dof_index_dg).index_storage_variants[2][cell] ==
                   internal::MatrixFreeFunctions::DoFInfo::IndexStorageVariants::contiguous,
                   ExcMessage("Invalid type " + std::to_string((int)matrixfree->get_dof_info(dof_index_dg).index_storage_variants[2][cell])));

            const typename DoFHandler<dim>::cell_iterator dcell=matrixfree->get_cell_iterator(cell, v, dof_index_dg);
            dcell->get_dof_indices(dof_indices);
            const internal::MatrixFreeFunctions::DoFInfo &dof_info
              = matrixfree->get_dof_info(dof_index_dg);
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
              if (dcell->at_boundary(f) && dcell->has_periodic_neighbor(f) == false)
                {
                  all_owned_faces(cell, f) = 0;
                  start_indices_on_neighbor(cell, f, v) =
                      dof_info.dof_indices_contiguous[2][cell*VectorizedArray<Number>::n_array_elements+v];
                  dirichlet_faces(cell, f, v) = 1;
                }
              else
                {
                  const typename DoFHandler<dim>::cell_iterator
                    neighbor=dcell->neighbor_or_periodic_neighbor(f);
                  AssertThrow(dcell->active(), ExcNotImplemented());
                  AssertThrow(neighbor->level() == dcell->level(), ExcNotImplemented());
                  if (neighbor->subdomain_id() == dcell->subdomain_id())
                    {
                      const auto id = std::make_pair(neighbor->level(),neighbor->index());
                      Assert(map_to_mf_numbering.find(id) != map_to_mf_numbering.end(),
                             ExcInternalError());
                      const unsigned int index = map_to_mf_numbering[id];
                      start_indices_on_neighbor(cell, f, v) = dof_info.dof_indices_contiguous[2][index];
                    }
                  else
                    {
                      start_indices_on_neighbor(cell, f, v) = 0;
                      all_owned_faces(cell, f) = 0;
                      std::array<types::global_dof_index,5> neighbor_data;
                      neighbor->get_dof_indices(dof_indices);
                      neighbor_data[0] = cell*VectorizedArray<Number>::n_array_elements + v;
                      neighbor_data[1] = dcell->has_periodic_neighbor(f) ?
                        dcell->periodic_neighbor_face_no(f) : dcell->neighbor_face_no(f);
                      neighbor_data[2] = dof_info.dof_indices_contiguous[2][cell*VectorizedArray<Number>::n_array_elements+v];
                      neighbor_data[3] = dof_indices[0];
                      neighbor_data[4] = f;

                      proc_neighbors[neighbor->subdomain_id()].push_back(neighbor_data);
                    }
                }
          }

      // find out how the neighbor wants us to send the data -> we sort by the
      // global dof id
      constexpr unsigned int dofs_per_face = Utilities::pow(fe_degree+1,dim-1);
      send_data_process.clear();
      send_data_dof_index.clear();
      send_data_face_index.clear();
      for (auto &it : proc_neighbors)
        {
          std::sort(it.second.begin(), it.second.end(),
                    [] (const std::array<types::global_dof_index,5> &a,
                        const std::array<types::global_dof_index,5> &b)
                    { if (a[3] < b[3])
                        return true;
                      else if (a[3] == b[3] && a[4] < b[4])
                        return true;
                      else
                        return false;
                    });
          send_data_process.emplace_back(it.first, it.second.size());
          for (unsigned int i=0; i<it.second.size(); ++i)
            {
              send_data_dof_index.emplace_back(it.second[i][2]);
              send_data_face_index.emplace_back(it.second[i][4]);
            }
        }
      import_values.clear();
      import_values.resize_fast(send_data_dof_index.size()*2*dofs_per_face);
      export_values.clear();
      export_values.resize_fast(send_data_dof_index.size()*2*dofs_per_face);
#pragma omp parallel
      {
#pragma omp for schedule (static)
        for (unsigned int i=0; i<import_values.size(); ++i)
          {
            import_values[i] = 0;
            export_values[i] = 0;
          }
      }

      // finally figure out where to read the imported face data from
      unsigned int offset = 0;
      for (auto &it : proc_neighbors)
        {
          std::sort(it.second.begin(), it.second.end(),
                    [] (const std::array<types::global_dof_index,5> &a,
                        const std::array<types::global_dof_index,5> &b)
                    { if (a[2] < b[2])
                        return true;
                      else if (a[2] == b[2] && a[1] < b[1])
                        return true;
                      else
                        return false;
                    });

          for (unsigned int i=0; i<it.second.size(); ++i)
            {
              const unsigned int cell = it.second[i][0]/VectorizedArray<Number>::n_array_elements;
              const unsigned int v = it.second[i][0]%VectorizedArray<Number>::n_array_elements;
              start_indices_auxiliary(cell, it.second[i][4], v) = offset;

              offset += 2*dofs_per_face;
            }
        }
      AssertDimension(offset, export_values.size());

      AssertThrow(matrixfree->get_mapping_info().cell_data[0].jacobians[0].size()<=1,
                  ExcNotImplemented());
      if (matrixfree->get_mapping_info().cell_data[0].jacobians[0].size())
        {
          coefficient.resize(1);
          const Tensor<2,dim,VectorizedArray<Number> >& inv_jac =
              matrixfree->get_mapping_info().cell_data[0].jacobians[0][0];
          const VectorizedArray<Number> my_jxw =
              matrixfree->get_mapping_info().cell_data[0].JxW_values[0];

          Tensor<2,dim,VectorizedArray<Number>> tmp = transpose(inv_jac)*inv_jac;
          for (unsigned int d=0; d<dim; ++d)
            coefficient[0][d] = my_jxw * tmp[d][d];
          for (unsigned int c=0,d=dim; c<dim; ++c)
            for (unsigned int e=c+1; e<dim; ++e, ++d)
              coefficient[0][d] = my_jxw * tmp[c][e];
        }
    }

    const MatrixFree<dim,Number> & get_matrix_free () const
    {
      return *matrixfree;
    }

    const AlignedVector<Tensor<1,(dim*(dim+1))/2,VectorizedArray<Number>>> &
    get_coefficients() const
    {
      return coefficient;
    }

    unsigned int get_dof_index_dg() const
    {
      return dof_index_dg;
    }

    VectorizedArray<Number>
    get_penalty(const unsigned int ,
                const unsigned int face) const
    {
      return Number(penalty_factor * (fe_degree+1) * (fe_degree+1)) * my_sigma[face/2];
    }

    types::global_dof_index m() const
    {
      return matrixfree->get_dof_info(dof_index_dg).vector_partitioner->size();
    }

    void initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec) const
    {
      // must set the vector to zero by a loop of the same form as in the loop
      // over the vector
      LinearAlgebra::distributed::Vector<Number> dummy;
      dummy.reinit(matrixfree->get_dof_info(dof_index_dg).vector_partitioner);
      vec.reinit(dummy, true);
      dummy.reinit(0);
#pragma omp parallel shared (vec)
      {
        const unsigned int n_cells = matrixfree->n_macro_cells();
#pragma omp for schedule (static)
        for (unsigned int cell=0; cell<n_cells; ++cell)
          for (unsigned int l=0; l<matrixfree->n_components_filled(cell); ++l)
            {
              const unsigned int index =
                  matrixfree->get_dof_info(dof_index_dg).dof_indices_contiguous[2][cell*VectorizedArray<Number>::n_array_elements+l];
            const unsigned int local_size = Utilities::pow(fe_degree+1,dim);
            AssertIndexRange(index + local_size, vec.local_size() + 1);
            std::memset(vec.begin()+index, 0, local_size*sizeof(Number));
          }
      }
    }

    void vmult(LinearAlgebra::distributed::Vector<Number> &dst,
               const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      vmult_with_merged_ops<0>(src,src,dst);
      //const double norm_src = src.l2_norm();
      //const double norm_dst = dst.l2_norm();
      //if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      //  std::cout << "norms mv " << norm_src << " " << norm_dst << std::endl;
      //if (dst.size()==128)
      //  {
      //  dst.print(std::cout);
      //  std::abort();
      //  }
    }

    void vmult_residual_and_restrict_to_cg
      (const LinearAlgebra::distributed::Vector<Number> &rhs,
       const LinearAlgebra::distributed::Vector<Number> &lhs,
       LinearAlgebra::distributed::Vector<Number> &restricted_residual) const
    {
      restricted_residual = 0;
      vmult_with_merged_ops<1>(lhs,rhs,restricted_residual);
      restricted_residual.compress(VectorOperation::add);
    }

    std::array<Number,4>
    vmult_with_cg_update (const Number alpha,
                          const Number beta,
                          const LinearAlgebra::distributed::Vector<Number> &r,
                          LinearAlgebra::distributed::Vector<Number> &q,
                          LinearAlgebra::distributed::Vector<Number> &p,
                          LinearAlgebra::distributed::Vector<Number> &x) const
    {
      using Simd = VectorizedArray<Number>;
      // update x vector with old content of p
      // update p vector according to beta formula
      // zero q vector (after having read p)
      const unsigned int local_size = r.local_size();
      Simd *arr_q = reinterpret_cast<Simd*>(q.begin());
      Simd *arr_p = reinterpret_cast<Simd*>(p.begin());
      Simd *arr_x = reinterpret_cast<Simd*>(x.begin());
      if (alpha == Number())
        {
          for (unsigned int i=0; i<local_size/Simd::n_array_elements; ++i)
            {
              arr_p[i] = arr_q[i];
            }
          for (unsigned int i=local_size/Simd::n_array_elements*Simd::n_array_elements; i<local_size; ++i)
            {
              p.local_element(i) = q.local_element(i);
            }
        }
      else
        {
          for (unsigned int i=0; i<local_size/Simd::n_array_elements; ++i)
            {
              arr_x[i] += alpha * arr_p[i];
              arr_p[i] = beta * arr_p[i] + arr_q[i];
            }
          for (unsigned int i=local_size/Simd::n_array_elements*Simd::n_array_elements; i<local_size; ++i)
            {
              x.local_element(i) += alpha * p.local_element(i);
              p.local_element(i) = beta * p.local_element(i) + q.local_element(i);
            }
        }
      std::array<Number,4> cg_sums = vmult_with_merged_ops<2>(p,r,q);
      Utilities::MPI::sum(ArrayView<const Number>(cg_sums.data(), 4),
                          q.get_mpi_communicator(),
                          ArrayView<Number>(cg_sums.data(), 4));
      return cg_sums;
    }

    void vmult_with_chebyshev_update
      (const JacobiTransformed<dim,fe_degree,Number> &jacobi_transformed,
       const LinearAlgebra::distributed::Vector<Number> &rhs,
       const unsigned int iteration_index,
       const Number factor1,
       const Number factor2,
       LinearAlgebra::distributed::Vector<Number> &solution,
       LinearAlgebra::distributed::Vector<Number> &solution_old,
       LinearAlgebra::distributed::Vector<Number> &/*temp_vector*/) const
    {
      if (iteration_index > 0)
        {
          vmult_with_merged_ops<3>(solution, rhs, solution_old, iteration_index,
                                   factor1, factor2, &jacobi_transformed);
          solution.swap(solution_old);
        }
      else
        {
          constexpr unsigned int dofs_per_cell = Utilities::pow(fe_degree+1,dim);
          FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> fe_eval(*matrixfree,dof_index_dg);
          for (unsigned int cell=0; cell<matrixfree->n_macro_cells(); ++cell)
            {
              fe_eval.reinit(cell);
              fe_eval.read_dof_values(rhs);
              jacobi_transformed.do_local_operation(cell, fe_eval.begin_dof_values(),
                                                    fe_eval.begin_dof_values());
              for (unsigned int i=0; i<dofs_per_cell; ++i)
                fe_eval.begin_dof_values()[i] = fe_eval.begin_dof_values()[i] * factor2;
              fe_eval.set_dof_values(solution);
            }
        }
    }

    // action = 0: usual matrix-vector product
    // action = 1: matrix-vector product merged with residual evaluation and
    //             restriction to CG space
    // action = 2: matrix-vector product merged with operations in CG solver
    // action = 3: matrix-vector product merged with Chebyshev smoother
    template <int action>
    std::array<Number,4>
    vmult_with_merged_ops(const LinearAlgebra::distributed::Vector<Number> &src,
                          const LinearAlgebra::distributed::Vector<Number> &rhs,
                          LinearAlgebra::distributed::Vector<Number> &dst,
                          const unsigned int iteration_index = 0,
                          const double factor1 = 0,
                          const double factor2 = 0,
                          const JacobiTransformed<dim,fe_degree,Number> *jacobi_transformed = nullptr) const
    {
      if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
        {
          constexpr unsigned int dofs_per_face = Utilities::pow(fe_degree+1,dim-1);
          std::vector<MPI_Request> requests(2*send_data_process.size());
          unsigned int offset = 0;
          for (unsigned int p=0; p<send_data_process.size(); ++p)
            {
              MPI_Irecv (&import_values[offset*2*dofs_per_face],
                         send_data_process[p].second*2*dofs_per_face*sizeof(Number),
                         MPI_BYTE,
                         send_data_process[p].first,
                         send_data_process[p].first + 47,
                         src.get_mpi_communicator(),
                         &requests[p]);
              offset += send_data_process[p].second;
            }
          AssertDimension(offset*2*dofs_per_face, import_values.size());

#pragma omp parallel shared (src)
#pragma omp for schedule (static)
          for (unsigned int offset = 0; offset < send_data_face_index.size(); ++offset)
            {
              const unsigned int face = send_data_face_index[offset];
              const unsigned int index = send_data_dof_index[offset];
              const unsigned int stride1 = Utilities::pow(fe_degree+1,(face/2+1)%dim);
              const unsigned int stride2 = Utilities::pow(fe_degree+1,(face/2+2)%dim);
              const unsigned int offset1 = ((face%2==0) ? 0 : fe_degree) * Utilities::pow(fe_degree+1,(face/2)%dim);
              const unsigned int offset2 = ((face%2==0) ? 1 : fe_degree-1) * Utilities::pow(fe_degree+1,(face/2)%dim);
              const Number w0 = (face % 2 == 0 ? 1. : -1.) * hermite_derivative_on_face[0];
              for (unsigned int i2=0, j=0; i2<(dim==2?1:(fe_degree+1)); ++i2)
                for (unsigned int i1=0; i1<fe_degree+1; ++i1, ++j)
                  {
                    const unsigned int my_index = (offset1+i2*stride2 + i1*stride1);
                    export_values[offset*2*dofs_per_face+2*j] =
                      src.local_element(index + my_index);
                    const unsigned int my_herm_index = (offset2+i2*stride2 + i1*stride1);
                    export_values[offset*2*dofs_per_face+2*j+1] =
                      w0 * (export_values[offset*2*dofs_per_face+2*j] -
                            src.local_element(index + my_herm_index));
                  }
            }

          offset = 0;
          for (unsigned int p=0; p<send_data_process.size(); ++p)
            {
              const unsigned int old_offset = offset*2*dofs_per_face;
              MPI_Isend (&export_values[old_offset],
                         send_data_process[p].second*2*dofs_per_face*sizeof(Number),
                         MPI_BYTE,
                         send_data_process[p].first,
                         src.get_partitioner()->this_mpi_process() + 47,
                         src.get_mpi_communicator(),
                         &requests[send_data_process.size()+p]);
              offset += send_data_process[p].second;
            }
          AssertDimension(offset*2*dofs_per_face, export_values.size());
          MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);
        }

      std::array<Number,4> result_cg = {};

#pragma omp parallel shared (dst, src, rhs)
      {
        std::array<Number,4> sums_cg = {};

        const unsigned int n_cells = matrixfree->n_macro_cells();

        constexpr unsigned int n_lanes = VectorizedArray<Number>::n_array_elements;
        constexpr unsigned int dofs_per_cell = Utilities::pow(fe_degree+1,dim);
        constexpr unsigned int dofs_per_face = Utilities::pow(fe_degree+1,dim-1);
        constexpr unsigned int dofs_per_plane = Utilities::pow(fe_degree+1,2);
        VectorizedArray<Number> vect_source[dofs_per_cell];
        VectorizedArray<Number> array[dofs_per_cell], array_2[(fe_degree < 5 ? 6 : (fe_degree+1))*dofs_per_face];
        VectorizedArray<Number> array_f[6][dofs_per_face], array_fd[6][dofs_per_face];
        const VectorizedArray<Number> *__restrict
          shape_values_eo = matrixfree->get_shape_info(dof_index_dg).shape_values_eo.begin();
        const VectorizedArray<Number> *__restrict
          shape_gradients_eo = matrixfree->get_shape_info(dof_index_dg).shape_gradients_collocation_eo.begin();
        const AlignedVector<Number> &quadrature_weight = matrixfree->get_mapping_info().cell_data[0].descriptor[0].quadrature_weights;
        AssertDimension(face_quadrature_weights.size(), dofs_per_face);
        constexpr unsigned int nn = fe_degree+1;
        constexpr unsigned int mid = nn/2;

#pragma omp for schedule (static)
        for (unsigned int cell = 0; cell<n_cells; ++cell)
          {
            const unsigned int *dof_indices =
              matrixfree->get_dof_info(dof_index_dg).dof_indices_contiguous[2].data()+cell*n_lanes;
            const Number* src_array = src.begin();

            const unsigned int n_lanes_filled =
                matrixfree->n_active_entries_per_cell_batch(cell);

            unsigned int read_idx = 0;
            for (unsigned int i2=0; i2<(dim>2 ? nn : 1); ++i2)
              {
                // x-direction
                VectorizedArray<Number> *__restrict in = array + i2*nn*nn;
                if (n_lanes_filled == n_lanes)
                  for (unsigned int i1=0; i1<nn; ++i1)
                    {
                      const unsigned int next_size = std::min(dofs_per_cell,
                                                              (i2*nn*nn+(i1+1)*nn+3)/4*4);
                      vectorized_load_and_transpose(next_size-read_idx,
                                                    src_array+read_idx,
                                                    dof_indices,
                                                    vect_source+read_idx);
                      read_idx = next_size;
                      apply_1d_matvec_kernel<nn, 1, 0, true, false, VectorizedArray<Number>>
                          (shape_values_eo, vect_source+i2*nn*nn+i1*nn, in+i1*nn);
                    }
                else
                  for (unsigned int i1=0; i1<nn; ++i1)
                    {
                      for (unsigned int i=0; i<nn; ++i)
                        vect_source[i2*nn*nn+i1*nn+i] = {};
                      for (unsigned int l=0; l<n_lanes_filled; ++l)
                        for (unsigned int i=0; i<nn; ++i)
                          vect_source[i2*nn*nn+i1*nn+i][l]
                              = src_array[dof_indices[l]+i2*nn*nn+i1*nn+i];
                      apply_1d_matvec_kernel<nn, 1, 0, true, false, VectorizedArray<Number>>
                          (shape_values_eo, vect_source+i2*nn*nn+i1*nn, in+i1*nn);
                    }
                // y-direction
                for (unsigned int i1=0; i1<nn; ++i1)
                  {
                    apply_1d_matvec_kernel<nn, nn, 0, true, false, VectorizedArray<Number>>
                      (shape_values_eo, in+i1, in+i1);
                  }
              }

            if (dim == 3)
              {
                for (unsigned int i2=0; i2<nn; ++i2)
                  {
                    // interpolate in z direction
                    for (unsigned int i1=0; i1<nn; ++i1)
                      {
                        array_f[4][i2*nn+i1] = array[i2*nn+i1];
                        array_fd[4][i2*nn+i1] = hermite_derivative_on_face*(array[i2*nn+i1] - array[nn*nn+i2*nn+i1]);
                        array_f[5][i2*nn+i1] = array[(nn-1)*nn*nn+i2*nn+i1];
                        array_fd[5][i2*nn+i1] = hermite_derivative_on_face*(array[(nn-2)*nn*nn+i2*nn+i1] - array[(nn-1)*nn*nn+i2*nn+i1]);
                        apply_1d_matvec_kernel<nn,nn*nn,0,true,false,VectorizedArray<Number>>
                          (shape_values_eo, array+i2*nn+i1, array+i2*nn+i1);
                      }

                    // interpolate onto x faces
                    for (unsigned int i1=0; i1<nn; ++i1)
                      {
                        VectorizedArray<Number> r0, r1, r2, r3;
                        {
                          const VectorizedArray<Number> t0 = array[i1*nn*nn+i2*nn];
                          const VectorizedArray<Number> t1 = array[i1*nn*nn+i2*nn+nn-1];
                          r0 = shape_values_on_face_eo[0] * (t0+t1);
                          r1 = shape_values_on_face_eo[nn-1] * (t0-t1);
                          r2 = shape_values_on_face_eo[nn] * (t0-t1);
                          r3 = shape_values_on_face_eo[2*nn-1] * (t0+t1);
                        }
                        for (unsigned int ind=1; ind<mid; ++ind)
                          {
                            const VectorizedArray<Number> t0 = array[i1*nn*nn+i2*nn+ind];
                            const VectorizedArray<Number> t1 = array[i1*nn*nn+i2*nn+nn-1-ind];
                            r0 += shape_values_on_face_eo[ind] * (t0+t1);
                            r1 += shape_values_on_face_eo[nn-1-ind] * (t0-t1);
                            r2 += shape_values_on_face_eo[nn+ind] * (t0-t1);
                            r3 += shape_values_on_face_eo[2*nn-1-ind] * (t0+t1);
                          }
                        if (nn%2 == 1)
                          {
                            r0 += shape_values_on_face_eo[mid] * array[i1*nn*nn+i2*nn+mid];
                            r3 += shape_values_on_face_eo[nn+mid] * array[i1*nn*nn+i2*nn+mid];
                          }
                        array_f[0][i1*nn+i2] = r0 + r1;
                        array_f[1][i1*nn+i2] = r0 - r1;
                        array_fd[0][i1*nn+i2] = r2 + r3;
                        array_fd[1][i1*nn+i2] = r2 - r3;
                      }
                  }
              }
            else
              {
                for (unsigned int i2=0; i2<nn; ++i2)
                  {
                    VectorizedArray<Number> r0, r1, r2, r3;
                    {
                      const VectorizedArray<Number> t0 = array[i2*nn];
                      const VectorizedArray<Number> t1 = array[i2*nn+nn-1];
                      r0 = shape_values_on_face_eo[0] * (t0+t1);
                      r1 = shape_values_on_face_eo[nn-1] * (t0-t1);
                      r2 = shape_values_on_face_eo[nn] * (t0-t1);
                      r3 = shape_values_on_face_eo[2*nn-1] * (t0+t1);
                    }
                    for (unsigned int ind=1; ind<mid; ++ind)
                      {
                        const VectorizedArray<Number> t0 = array[i2*nn+ind];
                        const VectorizedArray<Number> t1 = array[i2*nn+nn-1-ind];
                        r0 += shape_values_on_face_eo[ind] * (t0+t1);
                        r1 += shape_values_on_face_eo[nn-1-ind] * (t0-t1);
                        r2 += shape_values_on_face_eo[nn+ind] * (t0-t1);
                        r3 += shape_values_on_face_eo[2*nn-1-ind] * (t0+t1);
                      }
                    if (nn%2 == 1)
                      {
                        r0 += shape_values_on_face_eo[mid] * array[i2*nn+mid];
                        r3 += shape_values_on_face_eo[nn+mid] * array[i2*nn+mid];
                      }
                    array_f[0][i2] = r0 + r1;
                    array_f[1][i2] = r0 - r1;
                    array_fd[0][i2] = r2 + r3;
                    array_fd[1][i2] = r2 - r3;
                  }
              }

            // interpolate internal y values onto faces
            for (unsigned int i1=0; i1<(dim==3?nn:1); ++i1)
              {
                for (unsigned int i2=0; i2<nn; ++i2)
                  {
                    VectorizedArray<Number> r0, r1, r2, r3;
                    {
                      const VectorizedArray<Number> t0 = array[i1*nn*nn+i2];
                      const VectorizedArray<Number> t1 = array[i1*nn*nn+i2+(nn-1)*nn];
                      r0 = shape_values_on_face_eo[0] * (t0+t1);
                      r1 = shape_values_on_face_eo[nn-1] * (t0-t1);
                      r2 = shape_values_on_face_eo[nn] * (t0-t1);
                      r3 = shape_values_on_face_eo[2*nn-1] * (t0+t1);
                    }
                    for (unsigned int ind=1; ind<mid; ++ind)
                      {
                        const VectorizedArray<Number> t0 = array[i1*nn*nn+i2+ind*nn];
                        const VectorizedArray<Number> t1 = array[i1*nn*nn+i2+(nn-1-ind)*nn];
                        r0 += shape_values_on_face_eo[ind] * (t0+t1);
                        r1 += shape_values_on_face_eo[nn-1-ind] * (t0-t1);
                        r2 += shape_values_on_face_eo[nn+ind] * (t0-t1);
                        r3 += shape_values_on_face_eo[2*nn-1-ind] * (t0+t1);
                      }
                    if (nn%2 == 1)
                      {
                        r0 += shape_values_on_face_eo[mid] * array[i1*nn*nn+i2+mid*nn];
                        r3 += shape_values_on_face_eo[nn+mid] * array[i1*nn*nn+i2+mid*nn];
                      }
                    if (dim == 3)
                      {
                        array_f[2][i2*nn+i1] = r0 + r1;
                        array_f[3][i2*nn+i1] = r0 - r1;
                        array_fd[2][i2*nn+i1] = r2 + r3;
                        array_fd[3][i2*nn+i1] = r2 - r3;
                      }
                    else
                      {
                        array_f[2][i2] = r0 + r1;
                        array_f[3][i2] = r0 - r1;
                        array_fd[2][i2] = r2 + r3;
                        array_fd[3][i2] = r2 - r3;
                      }
                  }
              }

            // face integrals
            for (unsigned int f=0; f<2*dim; ++f)
              {
                // interpolate external values for faces
                const unsigned int stride1 = Utilities::pow(fe_degree+1,(f/2+1)%dim);
                const unsigned int stride2 = Utilities::pow(fe_degree+1,(f/2+2)%dim);
                const unsigned int offset1 = ((1-f%2)*fe_degree) * Utilities::pow(fe_degree+1,f/2);
                const unsigned int offset2 = ((1-f%2)*(fe_degree-2) + 1) * Utilities::pow(fe_degree+1,f/2);
                const VectorizedArray<Number> w0 = Number(int(1-2*(f%2)))*hermite_derivative_on_face;
                const unsigned int *index = &start_indices_on_neighbor[cell][f][0];
                if (all_owned_faces(cell, f) != 0)
                  for (unsigned int i2=0; i2<(dim==3 ? nn : 1); ++i2)
                    {
                      for (unsigned int i1=0; i1<nn; ++i1)
                        {
                          const unsigned int i=i2*nn+i1;
                          array_2[i].gather(src.begin()+(offset1+i2*stride2 + i1*stride1), index);
                          array_2[i+dofs_per_face].gather(src.begin()+(offset2+i2*stride2 + i1*stride1), index);
                          array_2[i+dofs_per_face] = w0 * (array_2[i+dofs_per_face] - array_2[i]);
                        }
                      apply_1d_matvec_kernel<nn, 1, 0, true, false, VectorizedArray<Number>>
                          (shape_values_eo, array_2+i2*nn, array_2+i2*nn);
                      apply_1d_matvec_kernel<nn, 1, 0, true, false, VectorizedArray<Number>>
                          (shape_values_eo, array_2+dofs_per_face+i2*nn, array_2+dofs_per_face+i2*nn);
                    }
                else
                  {
                    for (unsigned int v=0; v<n_lanes_filled; ++v)
                      {
                        const unsigned int my_index = start_indices_auxiliary(cell, f, v);
                        if (my_index != numbers::invalid_unsigned_int)
                          for (unsigned int i=0; i<dofs_per_face; ++i)
                            {
                              array_2[dofs_per_face+i][v] = import_values[my_index + 2*i+1];
                              array_2[i][v] = import_values[my_index + 2*i];
                            }
                        else if (dirichlet_faces(cell, f, v))
                          {
                            const unsigned int offset1 = ((f%2)*fe_degree) * Utilities::pow(fe_degree+1,f/2);
                            const unsigned int offset2 = ((f%2)*(fe_degree-2) + 1) * Utilities::pow(fe_degree+1,f/2);
                            for (unsigned int i2=0; i2<(dim==3 ? nn : 1); ++i2)
                              for (unsigned int i1=0; i1<nn; ++i1)
                                {
                                  const unsigned int i=i2*nn+i1;
                                  array_2[i][v] = -vect_source[offset1+i2*stride2 + i1*stride1][v];
                                  array_2[dofs_per_face+i][v] =
                                      w0[0] * (array_2[i][v]+
                                               vect_source[offset2+i2*stride2 + i1*stride1][v]);
                                }
                          }
                        else
                          for (unsigned int i2=0; i2<(dim==3 ? nn : 1); ++i2)
                            {
                              for (unsigned int i1=0; i1<nn; ++i1)
                                {
                                  const unsigned int i=i2*nn+i1;
                                  array_2[i][v] = src.local_element(index[v]+offset1+i2*stride2 + i1*stride1);
                                  array_2[i+dofs_per_face][v] =
                                      src.local_element(index[v]+offset2+i2*stride2 + i1*stride1);
                                  array_2[i+dofs_per_face][v] = w0[0] *
                                      (array_2[i+dofs_per_face][v] - array_2[i][v]);
                                }
                            }
                      }
                    for (unsigned int i2=0; i2<(dim==3 ? nn : 1); ++i2)
                      apply_1d_matvec_kernel<nn, 1, 0, true, false, VectorizedArray<Number>>
                        (shape_values_eo, array_2+i2*nn, array_2+i2*nn);
                    for (unsigned int i2=0; i2<(dim==3 ? nn : 1); ++i2)
                      apply_1d_matvec_kernel<nn, 1, 0, true, false, VectorizedArray<Number>>
                        (shape_values_eo, array_2+dofs_per_face+i2*nn, array_2+dofs_per_face+i2*nn);
                  }
                for (unsigned int i1=0; i1<(dim==3 ? nn : 0); ++i1)
                  apply_1d_matvec_kernel<nn, nn, 0, true, false, VectorizedArray<Number>>
                    (shape_values_eo, array_2+dofs_per_face+i1, array_2+dofs_per_face+i1);
                for (unsigned int i1=0; i1<(dim==3 ? nn : 0); ++i1)
                  apply_1d_matvec_kernel<nn, nn, 0, true, false, VectorizedArray<Number>>
                    (shape_values_eo, array_2+i1, array_2+i1);
                for (unsigned int i1=0; i1<(dim==3 ? nn : 1); ++i1)
                  apply_1d_matvec_kernel<nn, 1, 1, true, false, VectorizedArray<Number>>
                    (shape_gradients_eo, array_2+i1*nn, array_2+2*dofs_per_face+i1*nn);
                for (unsigned int i1=0; i1<(dim==3 ? nn : 0); ++i1)
                  apply_1d_matvec_kernel<nn, nn, 1, true, false, VectorizedArray<Number>>
                    (shape_gradients_eo, array_2+i1, array_2+3*dofs_per_face+i1);
                for (unsigned int i1=0; i1<(dim==3 ? nn : 1); ++i1)
                  apply_1d_matvec_kernel<nn, 1, 1, true, false, VectorizedArray<Number>>
                    (shape_gradients_eo, array_f[f]+i1*nn, array_2+4*dofs_per_face+i1*nn);
                for (unsigned int i1=0; i1<(dim==3 ? nn : 0); ++i1)
                  apply_1d_matvec_kernel<nn, nn, 1, true, false, VectorizedArray<Number>>
                    (shape_gradients_eo, array_f[f]+i1, array_2+5*dofs_per_face+i1);

                VectorizedArray<Number> sigma = get_penalty(cell, f);
                const Tensor<1,dim,VectorizedArray<Number>> &jac1 = f%2==0 ? normal_jac1[f/2] : normal_jac2[f/2];
                Tensor<1,dim,VectorizedArray<Number>> jac2 = f%2==0 ? normal_jac2[f/2] : normal_jac1[f/2];
                for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
                  if (dirichlet_faces(cell,f,v))
                    {
                      for (unsigned int d=0; d<dim; ++d)
                        jac2[d][v] = -jac2[d][v];
                    }
                for (unsigned int q=0; q<dofs_per_face; ++q)
                  {
                    VectorizedArray<Number> grad1 = array_fd[f][q]  * jac1[dim-1];
                    grad1 += array_2[4*dofs_per_face+q] * jac1[0];
                    VectorizedArray<Number> grad2 = array_2[dofs_per_face+q] * jac2[dim-1];
                    grad2 += array_2[2*dofs_per_face+q] * jac2[0];
                    if (dim==3)
                      {
                        grad1 += array_2[5*dofs_per_face+q] * jac1[1];
                        grad2 += array_2[3*dofs_per_face+q] * jac2[1];
                      }
                    VectorizedArray<Number> jump_value = (array_f[f][q] - array_2[q]);
                    const VectorizedArray<Number> weight = -face_quadrature_weights[q] * face_jxw[f/2];
                    array_f[f][q] = (0.5 * (grad1 - grad2) - jump_value * sigma) * weight;
                    jump_value *= make_vectorized_array<Number>(0.5);
                    array_fd[f][q] = jump_value * weight * jac1[dim-1];
                    array_2[q] = jump_value * weight * jac1[0];
                    if (dim==3)
                      array_2[dofs_per_face+q] = jump_value * weight * jac1[1];
                  }
                for (unsigned int i1=0; i1<(dim==3?nn:0); ++i1)
                  apply_1d_matvec_kernel<nn, nn, 1, false, true, VectorizedArray<Number>>
                    (shape_gradients_eo, array_2+dofs_per_face+i1,
                     array_f[f] + i1, array_f[f] + i1);
                for (unsigned int i1=0; i1<(dim==3?nn:1); ++i1)
                  apply_1d_matvec_kernel<nn, 1, 1, false, true, VectorizedArray<Number>>
                    (shape_gradients_eo, array_2+i1*nn,
                     array_f[f] + i1*nn, array_f[f] + i1*nn);
              }

            /*
            for (unsigned int v=0; v<4; ++v)
              for (unsigned int f=0; f<6; ++f)
                {
                  for (unsigned int i=0; i<dofs_per_face; ++i)
                    std::cout << array_f[f][i][v] << " ";
                  std::cout << std::endl;
                  for (unsigned int i=0; i<dofs_per_face; ++i)
                    std::cout << array_fd[f][i][v] << " ";
                  std::cout << std::endl << std::endl;
                }
            */

            if (dim==3)
              for (unsigned int i2=0; i2<nn*nn; ++i2)
                apply_1d_matvec_kernel<nn,nn*nn,1,true,false,VectorizedArray<Number>>
                  (shape_gradients_eo, array+i2, array_2+i2);

            for (unsigned int i2=0; i2<(dim==3 ? nn : 1); ++i2)
              {
                const unsigned int offset = i2*dofs_per_plane;
                VectorizedArray<Number> *array_ptr = array + offset;
                VectorizedArray<Number> *array_2_ptr = array_2 + offset;

                VectorizedArray<Number> outy[dofs_per_plane];
                // y-derivative
                for (unsigned int i1=0; i1<nn; ++i1) // loop over x layers
                  {
                    apply_1d_matvec_kernel<nn, nn, 1, true, false, VectorizedArray<Number>>
                      (shape_gradients_eo, array_ptr+i1, outy+i1);
                  }

                // x-derivative
                for (unsigned int i1=0; i1<nn; ++i1) // loop over y layers
                  {
                    VectorizedArray<Number> outx[nn];
                    apply_1d_matvec_kernel<nn, 1, 1, true, false, VectorizedArray<Number>>
                      (shape_gradients_eo, array_ptr+i1*nn, outx);

                    for (unsigned int i=0; i<nn; ++i)
                      {
                        const VectorizedArray<Number> weight =
                          make_vectorized_array(quadrature_weight[i2*nn*nn+i1*nn+i]);
                        if (dim==2)
                          {
                            VectorizedArray<Number> t0 = outy[i1*nn+i]*coefficient[0][2] + outx[i]*coefficient[0][0];
                            VectorizedArray<Number> t1 = outy[i1*nn+i]*coefficient[0][1] + outx[i]*coefficient[0][2];
                            outx[i] = t0 * weight;
                            outy[i1*nn+i] = t1 * weight;
                          }
                        else if (dim==3)
                          {
                            VectorizedArray<Number> t0 = outy[i1*nn+i]*coefficient[0][3]+array_2_ptr[i1*nn+i]*coefficient[0][4] + outx[i]*coefficient[0][0];
                            VectorizedArray<Number> t1 = outy[i1*nn+i]*coefficient[0][1]+array_2_ptr[i1*nn+i]*coefficient[0][5] + outx[i]*coefficient[0][3];
                            VectorizedArray<Number> t2 = outy[i1*nn+i]*coefficient[0][5]+array_2_ptr[i1*nn+i]*coefficient[0][2] + outx[i]*coefficient[0][4];
                            outx[i] = t0 * weight;
                            outy[i1*nn+i] = t1 * weight;
                            array_2_ptr[i1*nn+i] = t2 * weight;
                          }
                      }
                    VectorizedArray<Number> array_face[4];
                    array_face[0] = array_f[0][i2*nn+i1]+array_f[1][i2*nn+i1];
                    array_face[1] = array_f[0][i2*nn+i1]-array_f[1][i2*nn+i1];
                    array_face[2] = array_fd[0][i2*nn+i1]+array_fd[1][i2*nn+i1];
                    array_face[3] = array_fd[0][i2*nn+i1]-array_fd[1][i2*nn+i1];
                    apply_1d_matvec_kernel<nn,1,1,false,false,VectorizedArray<Number>,VectorizedArray<Number>,false,2>
                      (shape_gradients_eo, outx, array_ptr+i1*nn,
                       nullptr, shape_values_on_face_eo.begin(), array_face);
                  }

                for (unsigned int i=0; i<nn; ++i)
                  {
                    const unsigned int i1 = dim==3 ? i*nn+i2 : i;
                    VectorizedArray<Number> array_face[4];
                    array_face[0] = array_f[2][i1]+array_f[3][i1];
                    array_face[1] = array_f[2][i1]-array_f[3][i1];
                    array_face[2] = array_fd[2][i1]+array_fd[3][i1];
                    array_face[3] = array_fd[2][i1]-array_fd[3][i1];
                    apply_1d_matvec_kernel<nn,nn,1,false,true,VectorizedArray<Number>,VectorizedArray<Number>,false,2>
                      (shape_gradients_eo, outy+i, array_ptr+i,
                       array_ptr+i, shape_values_on_face_eo.begin(), array_face);
                  }
              }
            if (dim == 3)
              {
                for (unsigned int i2=0; i2<dofs_per_face; ++i2)
                  {
                    VectorizedArray<Number> array_face[4];
                    array_face[0] = array_f[4][i2]+array_f[5][i2];
                    array_face[1] = array_f[4][i2]-array_f[5][i2];
                    array_face[2] = array_fd[4][i2]+array_fd[5][i2];
                    array_face[3] = array_fd[4][i2]-array_fd[5][i2];
                    apply_1d_matvec_kernel<nn,nn*nn,1,false,true,VectorizedArray<Number>,VectorizedArray<Number>,false,2>
                      (shape_gradients_eo, array_2+i2, array+i2,
                       array+i2, shape_values_on_face_eo.begin(), array_face);

                    apply_1d_matvec_kernel<nn,nn*nn,0,false,false,VectorizedArray<Number>,VectorizedArray<Number>>
                      (shape_values_eo, array+i2, array+i2);
                  }
              }

            read_idx = 0;
            for (unsigned int i2=0; i2< (dim>2 ? nn : 1); ++i2)
              {
                const unsigned int offset = i2*dofs_per_plane;
                // y-direction
                for (unsigned int i1=0; i1<nn; ++i1)
                  apply_1d_matvec_kernel<nn, nn, 0, false, false, VectorizedArray<Number>>
                    (shape_values_eo, array+offset+i1, array+offset+i1);
                for (unsigned int i1=0; i1<nn; ++i1)
                  {
                    apply_1d_matvec_kernel<nn, 1, 0, false, false, VectorizedArray<Number>>
                        (shape_values_eo, array+offset+i1*nn, array+offset+i1*nn);
                    if ((action == 0 || action == 2) && n_lanes_filled ==
                        VectorizedArray<Number>::n_array_elements)
                      {
                        unsigned int next_size = (i2*nn*nn+(i1+1)*nn)/4*4;
                        if (i2*nn*nn+i1*nn==dofs_per_cell-nn)
                          next_size = dofs_per_cell;
                        vectorized_transpose_and_store(false,
                                                       next_size-read_idx,
                                                       array+read_idx,
                                                       dof_indices,
                                                       dst.begin()+read_idx);
                        read_idx = next_size;
                      }
                  }
              }
            if ((action == 0 || action == 2) && n_lanes_filled
                < VectorizedArray<Number>::n_array_elements)
              write_dg(n_lanes_filled,
                       false, dofs_per_cell, array, dof_indices,
                       dst.begin());
            if (action == 1)
              {
                read_dg(n_lanes_filled,
                        dofs_per_cell, rhs.begin(),
                        dof_indices, array_2);
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  array[i] = array_2[i] - array[i];
                local_basis_transformer->template apply<true>(array, array);
                distribute_local_to_global_compressed<dim,fe_degree,Number>
                    (dst, op_fe->get_compressed_dof_indices(),
                     op_fe->get_all_indices_uniform(), cell, array);
              }
            else if (action == 2)
              {
                for (unsigned int v=0; v<n_lanes_filled; ++v)
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                      sums_cg[0] += array[i][v] * vect_source[i][v];
                      sums_cg[1] += rhs.local_element(dof_indices[v]+i) *
                                    rhs.local_element(dof_indices[v]+i);
                      sums_cg[2] += array[i][v] *
                                    rhs.local_element(dof_indices[v]+i);
                      sums_cg[3] += array[i][v] * array[i][v];
                    }
              }
            else if (action == 3)
              {
                read_dg(n_lanes_filled,
                        dofs_per_cell, rhs.begin(),
                        dof_indices, array_2);
                for (unsigned int i=0; i<dofs_per_cell; ++i)
                  array[i] = array_2[i] - array[i];
                Assert(jacobi_transformed != nullptr, ExcInternalError());
                jacobi_transformed->do_local_operation(cell, array, array);
                const Number factor1_plus_1 = 1. + factor1;
                if (iteration_index == 1)
                  {
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                      array[i] = factor2 * array[i] +
                                 factor1_plus_1*vect_source[i];
                  }
                else
                  {
                    read_dg(n_lanes_filled,
                            dofs_per_cell, dst.begin(),
                            dof_indices, array_2);
                    for (unsigned int i=0; i<dofs_per_cell; ++i)
                      array[i] = factor2 * array[i] +
                                 factor1_plus_1*vect_source[i] -
                                 factor1 * array_2[i];
                  }
                write_dg(n_lanes_filled,
                         false, dofs_per_cell, array, dof_indices,
                         dst.begin());
              }
          }
        //#pragma omp critical
        for (unsigned int i=0; i<4; ++i)
          result_cg[i] += sums_cg[i];
      }
      return result_cg;
    }

    void prolongate_add_cg_to_dg(LinearAlgebra::distributed::Vector<Number> &dst,
                                 const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      src.update_ghost_values();
      FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> fe_eval(*matrixfree, dof_index_dg);
      for (unsigned int cell=0; cell<matrixfree->n_macro_cells(); ++cell)
        {
          fe_eval.reinit(cell);
          read_dof_values_compressed<dim,fe_degree,Number>
            (src, op_fe->get_compressed_dof_indices(),
             op_fe->get_all_indices_uniform(), cell, fe_eval.begin_dof_values());
          local_basis_transformer->template apply<false>(fe_eval.begin_dof_values(),
                                                         fe_eval.begin_dof_values());
          fe_eval.distribute_local_to_global(dst);
        }
      src.zero_out_ghosts();
    }

    void add_face_integral_to_array(const unsigned int cell,
                                    const unsigned int face,
                                    const VectorizedArray<Number> *in_array,
                                    VectorizedArray<Number> *out_array) const
    {
      VectorizedArray<Number> sigmaF = get_penalty(cell, face);

      VectorizedArray<Number> factor_boundary;
      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
        // interior face
        if (dirichlet_faces(cell,face,v) == 0)
          factor_boundary[v] = 0.5;
      // Dirichlet boundary
        else
          factor_boundary[v] = 1.0;

      constexpr unsigned int n_q_points = Utilities::pow(fe_degree+1,dim-1);
      VectorizedArray<Number> temp1[2*n_q_points];
      VectorizedArray<Number> values[n_q_points];
      VectorizedArray<Number> grads[dim*n_q_points];
      VectorizedArray<Number> scratch[2*n_q_points];
      internal::FEFaceNormalEvaluationImpl<dim,
                                           fe_degree,
                                           1,
                                           VectorizedArray<Number>>::
        template interpolate<true, false>(
          matrixfree->get_shape_info(dof_index_dg), in_array, temp1, true, face);
      internal::FEFaceEvaluationImpl<true,dim,fe_degree,fe_degree+1,1,
        VectorizedArray<Number>>::evaluate_in_face(matrixfree->get_shape_info(dof_index_dg),
                                                   temp1,
                                                   values,
                                                   grads,
                                                   scratch,
                                                   true,
                                                   true,
                                                   GeometryInfo<dim>::max_children_per_cell);

      for (unsigned int q=0; q<n_q_points; ++q)
        {
          const VectorizedArray<Number> average_value = values[q] * factor_boundary;
          const Tensor<1,dim,VectorizedArray<Number>> &normal_times_jac =
              face%2 ? normal_jac2[face/2] : normal_jac1[face/2];
          VectorizedArray<Number> normal_der = grads[q] * normal_times_jac[0];
          for (unsigned int d=1; d<dim; ++d)
            normal_der += grads[q+d*n_q_points] * normal_times_jac[d];
          const VectorizedArray<Number> weight = face_quadrature_weights[q] * face_jxw[face/2];
          const VectorizedArray<Number> average_valgrad = (2.0 * average_value * sigmaF  -
              normal_der * factor_boundary) * weight;
          for (unsigned int d=0; d<dim; ++d)
            grads[q+d*n_q_points] = -average_value * weight * normal_times_jac[d];
          values[q] = average_valgrad;
        }
      internal::FEFaceEvaluationImpl<
        true,
        dim,
        fe_degree,
        fe_degree+1,
        1,
        VectorizedArray<Number>>::integrate_in_face(matrixfree->get_shape_info(dof_index_dg),
                                                    temp1,
                                                    values,
                                                    grads,
                                                    scratch,
                                                    true,
                                                    true,
                                                    GeometryInfo<dim>::max_children_per_cell);
      internal::FEFaceNormalEvaluationImpl<dim,
                                           fe_degree,
                                           1,
                                           VectorizedArray<Number>>::
        template interpolate<false, true>(
          matrixfree->get_shape_info(dof_index_dg), temp1, out_array, true, face);
    }

  private:
    std::shared_ptr<const MatrixFree<dim,Number>> matrixfree;
    unsigned int dof_index_dg;
    Table<3,unsigned int> start_indices_on_neighbor;
    std::shared_ptr<const Utilities::MPI::Partitioner> main_partitioner;
    mutable AlignedVector<Number> import_values;
    mutable AlignedVector<Number> export_values;
    Table<2,unsigned char> all_owned_faces;
    Table<3,unsigned char> dirichlet_faces;
    Table<3,unsigned int> start_indices_auxiliary;
    std::vector<std::pair<unsigned int, unsigned int>> send_data_process;
    std::vector<unsigned int> send_data_dof_index;
    std::vector<unsigned char> send_data_face_index;

    AlignedVector<VectorizedArray<Number> > shape_values_on_face_eo;
    VectorizedArray<Number> hermite_derivative_on_face;
    std::array<Tensor<1,dim,VectorizedArray<Number>>,dim> normal_jac1, normal_jac2, normal_vector;
    std::array<VectorizedArray<Number>,dim> face_jxw;
    std::array<VectorizedArray<Number>,dim> my_sigma;
    AlignedVector<Tensor<1,(dim*(dim+1))/2,VectorizedArray<Number>>> coefficient;
    AlignedVector<Number> face_quadrature_weights;
    std::shared_ptr<LocalBasisTransformer<dim,1,fe_degree,Number>> local_basis_transformer;
    const LaplaceOperator<dim,fe_degree,Number> *op_fe;
  };



  template <int dim, int fe_degree, typename Number>
  class JacobiTransformed
  {
  public:
    JacobiTransformed (const LaplaceOperatorCompactCombine<dim,fe_degree,Number> &laplace)
      :
      mf(laplace.get_matrix_free()),
      dof_index_dg(laplace.get_dof_index_dg()),
      local_basis_transformer(laplace.get_matrix_free(), laplace.get_dof_index_dg())
    {
      local_compute_diagonals(laplace);
    }

    types::global_dof_index m() const
    {
      return mf.get_dof_info(dof_index_dg).vector_partitioner->size();
    }

    void vmult(LinearAlgebra::distributed::Vector<Number> &dst,
               const LinearAlgebra::distributed::Vector<Number> &src) const
    {
      const unsigned int dofs_per_cell = Utilities::pow(fe_degree+1,dim);
      tmp_array.resize_fast(dofs_per_cell);
      for (unsigned int cell=0; cell<mf.n_macro_cells(); ++cell)
        {
          const unsigned int n_lanes_filled =
              mf.n_active_entries_per_cell_batch(cell);
          if (mf.get_dof_info(dof_index_dg).index_storage_variants[2][cell] ==
              internal::MatrixFreeFunctions::DoFInfo::IndexStorageVariants::contiguous)
            {
              const unsigned int *indices = mf.get_dof_info(dof_index_dg).dof_indices_contiguous[2].data() + cell*VectorizedArray<Number>::n_array_elements;
              read_dg(n_lanes_filled,
                      dofs_per_cell, src.begin(),
                      indices, tmp_array.begin());
              do_local_operation(cell, tmp_array.begin(), tmp_array.begin());
              write_dg(n_lanes_filled, false, dofs_per_cell,
                       tmp_array.begin(), indices, dst.begin());
            }
          else
            AssertThrow(false, ExcNotImplemented());
        }
      /*
      const double nrm_src = src.l2_norm();
      const double nrm_dst = dst.l2_norm();
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        {
          std::cout << std::setprecision(12);
          std::cout << "norms: " << nrm_src << " " << nrm_dst << std::endl;
        }
        */
    }

    void do_local_operation(const unsigned int cell,
                            const VectorizedArray<Number> *input,
                            VectorizedArray<Number> *output) const
    {
      local_basis_transformer.template apply<true>(input, output);
      const VectorizedArray<Number> *my_diagonal_entries =
        diagonal_entries.begin() + pointer_to_diagonal[cell];
      for (unsigned int q=0; q<mf.get_shape_info(dof_index_dg).n_q_points; ++q)
        output[q] *= my_diagonal_entries[q];
      local_basis_transformer.template apply<false>(output, output);
    }

  private:
    void local_compute_diagonals(const LaplaceOperatorCompactCombine<dim,fe_degree,Number> &laplace)
    {
      pointer_to_diagonal.resize(mf.n_macro_cells(), 0);
      const unsigned int dofs_per_cell = Utilities::pow(fe_degree+1, dim);
      for (unsigned int c=1; c<mf.n_macro_cells(); ++c)
        // same geometry as before, constant coefficient
//        if (mf.get_mapping_info().cell_data[0].data_index_offsets[c] ==
//            mf.get_mapping_info().cell_data[0].data_index_offsets[c-1])
//          pointer_to_diagonal[c] = pointer_to_diagonal[c-1];
//        else
          pointer_to_diagonal[c] = pointer_to_diagonal[c-1] + dofs_per_cell;

      if (pointer_to_diagonal.size() > 0)
        diagonal_entries.resize_fast(pointer_to_diagonal.back() + dofs_per_cell);

      FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> phi (mf, dof_index_dg);

      AlignedVector<VectorizedArray<Number> > out_array(phi.dofs_per_cell);
      for (unsigned int cell=0; cell<mf.n_macro_cells(); ++cell)
        if (cell == 0 || pointer_to_diagonal[cell] > pointer_to_diagonal[cell-1])
          {
            phi.reinit(cell);
            const Tensor<1,(dim*dim+dim)/2,VectorizedArray<Number> > *coefficient =
              laplace.get_coefficients().begin()+
              mf.get_mapping_info().cell_data[0].data_index_offsets[cell];
            const unsigned int n_q_points = phi.static_n_q_points;
            for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
              {
                for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
                  phi.begin_dof_values()[j] = VectorizedArray<Number>();
                phi.begin_dof_values()[i] = 1.;
                local_basis_transformer.template apply<false>(phi.begin_dof_values(),
                                                              phi.begin_dof_values());
                phi.evaluate (false,true);
                VectorizedArray<Number> *phi_grads = phi.begin_gradients();
                if (mf.get_mapping_info().cell_type[cell] > internal::MatrixFreeFunctions::affine)
                  for (unsigned int q=0; q<n_q_points; ++q)
                    {
                      if (dim == 3)
                        {
                          VectorizedArray<Number> t0 = phi_grads[q]*coefficient[q][0] + phi_grads[q+n_q_points]*coefficient[q][3]+phi_grads[q+2*n_q_points]*coefficient[q][4];
                          VectorizedArray<Number> t1 = phi_grads[q]*coefficient[q][3] + phi_grads[q+n_q_points]*coefficient[q][1]+phi_grads[q+2*n_q_points]*coefficient[q][5];
                          VectorizedArray<Number> t2 = phi_grads[q]*coefficient[q][4] + phi_grads[q+n_q_points]*coefficient[q][5]+phi_grads[q+2*n_q_points]*coefficient[q][2];
                          phi_grads[q] = t0;
                          phi_grads[q+n_q_points] = t1;
                          phi_grads[q+2*n_q_points] = t2;
                        }
                      else if (dim == 2)
                        {
                          VectorizedArray<Number> t0 = phi_grads[q]*coefficient[q][0] + phi_grads[q+n_q_points]*coefficient[q][2];
                          VectorizedArray<Number> t1 = phi_grads[q]*coefficient[q][2] + phi_grads[q+n_q_points]*coefficient[q][1];
                          phi_grads[q] = t0;
                          phi_grads[q+n_q_points] = t1;
                        }
                      else if (dim == 1)
                        {
                          phi_grads[q] *= coefficient[q][0];
                        }
                    }
                else
                  for (unsigned int q=0; q<n_q_points; ++q)
                    {
                      const Number weight = mf.get_mapping_info().cell_data[0].descriptor[0].quadrature_weights[q];
                      if (dim == 3)
                        {
                          VectorizedArray<Number> t0 = phi_grads[q]*coefficient[0][0] + phi_grads[q+n_q_points]*coefficient[0][3]+phi_grads[q+2*n_q_points]*coefficient[0][4];
                          VectorizedArray<Number> t1 = phi_grads[q]*coefficient[0][3] + phi_grads[q+n_q_points]*coefficient[0][1]+phi_grads[q+2*n_q_points]*coefficient[0][5];
                          VectorizedArray<Number> t2 = phi_grads[q]*coefficient[0][4] + phi_grads[q+n_q_points]*coefficient[0][5]+phi_grads[q+2*n_q_points]*coefficient[0][2];
                          phi_grads[q] = t0 * weight;
                          phi_grads[q+n_q_points] = t1 * weight;
                          phi_grads[q+2*n_q_points] = t2 * weight;
                        }
                      else if (dim == 2)
                        {
                          VectorizedArray<Number> t0 = phi_grads[q]*coefficient[0][0] + phi_grads[q+n_q_points]*coefficient[0][2];
                          VectorizedArray<Number> t1 = phi_grads[q]*coefficient[0][2] + phi_grads[q+n_q_points]*coefficient[0][1];
                          phi_grads[q] = t0 * weight;
                          phi_grads[q+n_q_points] = t1 * weight;
                        }
                      else if (dim == 1)
                        {
                          phi_grads[q] *= coefficient[q][0] * weight;
                        }
                    }
                phi.integrate (false,true,out_array.begin());

                for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
                  {
                    laplace.add_face_integral_to_array(cell, face,
                                                       phi.begin_dof_values(),
                                                       out_array.begin());
                  }
                local_basis_transformer.template apply<true>(out_array.begin(),
                                                             out_array.begin());
                //for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
                //  std::cout << out_array[j][0] << " ";
                //std::cout << std::endl;
                diagonal_entries[pointer_to_diagonal[cell]+i] = 1./out_array[i];
              }
          }
      if (false)
        {
          std::cout << "diag: ";
          for (unsigned int i=0; i<diagonal_entries.size(); )
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                for (unsigned int v=0; v<mf.n_components_filled(i/dofs_per_cell); ++v)
                  std::cout << 1./diagonal_entries[i+j][v] << " ";
              i += dofs_per_cell;
            }
          std::cout << std::endl;
        }
    }

    const MatrixFree<dim,Number> &mf;
    const unsigned int dof_index_dg;
    LocalBasisTransformer<dim,0,fe_degree,Number> local_basis_transformer;
    AlignedVector<VectorizedArray<Number> > diagonal_entries;
    std::vector<unsigned int> pointer_to_diagonal;
    mutable AlignedVector<VectorizedArray<Number> > tmp_array;
  };





}

#endif
