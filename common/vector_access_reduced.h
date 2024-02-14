
#ifndef vector_access_reduced_h
#define vector_access_reduced_h

#include <deal.II/base/vectorization.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/matrix_free/fe_evaluation.h>

template <int dim, int fe_degree, int n_components, typename Number>
void
read_dof_values_compressed(const dealii::LinearAlgebra::distributed::Vector<Number> &vec_in,
                           const std::vector<unsigned int>  &compressed_indices,
                           const std::vector<unsigned char> &all_indices_unconstrained,
                           const unsigned int                cell_no,
                           dealii::VectorizedArray<Number>  *dof_values)
{
  dealii::LinearAlgebra::distributed::Vector<Number> &vec =
    const_cast<dealii::LinearAlgebra::distributed::Vector<Number> &>(vec_in);
  AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) *
                     dealii::VectorizedArray<Number>::size(),
                   compressed_indices.size());
  using VectorizedArrayType            = dealii::VectorizedArray<Number>;
  constexpr unsigned int n_lanes       = VectorizedArrayType::size();
  constexpr unsigned int n_q_points_1d = fe_degree + 1;
  const unsigned int    *cell_indices =
    compressed_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *cell_unconstrained =
    all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
  constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(n_q_points_1d, dim);
  dealii::internal::VectorReader<Number, VectorizedArrayType> reader;

  for (unsigned int i2 = 0, compressed_i2 = 0, offset_i2 = 0; i2 < (dim == 3 ? (fe_degree + 1) : 1);
       ++i2)
    {
      bool all_unconstrained = true;
      for (unsigned int i = 0; i < 9; ++i)
        if (cell_unconstrained[9 * compressed_i2 + i] == 0)
          all_unconstrained = false;
      if (n_components == 1 && fe_degree < 8 && all_unconstrained)
        {
          const unsigned int *indices = cell_indices + 9 * n_lanes * compressed_i2;
          // first line
          reader.process_dof_gather(indices,
                                    vec,
                                    offset_i2,
                                    vec.begin() + offset_i2,
                                    dof_values[0],
                                    std::integral_constant<bool, true>());
          indices += n_lanes;
          constexpr unsigned int n_regular = (fe_degree - 1) * n_components / 4 * 4;
          dealii::vectorized_load_and_transpose(n_regular,
                                                vec.begin() +
                                                  offset_i2 * (fe_degree - 1) * n_components,
                                                indices,
                                                dof_values + 1);
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            reader.process_dof_gather(indices,
                                      vec,
                                      offset_i2 * (fe_degree - 1) * n_components + i0,
                                      vec.begin() + offset_i2 * (fe_degree - 1) * n_components + i0,
                                      dof_values[1 + i0],
                                      std::integral_constant<bool, true>());
          indices += n_lanes;
          reader.process_dof_gather(indices,
                                    vec,
                                    offset_i2,
                                    vec.begin() + offset_i2,
                                    dof_values[fe_degree],
                                    std::integral_constant<bool, true>());
          indices += n_lanes;

          // inner part
          VectorizedArrayType tmp[fe_degree > 1 ? (fe_degree - 1) * (fe_degree - 1) : 1];
          dealii::vectorized_load_and_transpose(
            n_regular, vec.begin() + offset_i2 * (fe_degree - 1) * n_components, indices, tmp);
          for (unsigned int i0 = 0; i0 < n_regular; ++i0)
            dof_values[(fe_degree + 1) * (i0 + 1)] = tmp[i0];
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            reader.process_dof_gather(indices,
                                      vec,
                                      offset_i2 * (fe_degree - 1) + i0,
                                      vec.begin() + offset_i2 * (fe_degree - 1) + i0,
                                      dof_values[(fe_degree + 1) * (i0 + 1)],
                                      std::integral_constant<bool, true>());
          indices += n_lanes;

          constexpr unsigned int n2_regular =
            (fe_degree - 1) * (fe_degree - 1) * n_components / 4 * 4;
          dealii::vectorized_load_and_transpose(n2_regular,
                                                vec.begin() + offset_i2 * (fe_degree - 1) *
                                                                (fe_degree - 1) * n_components,
                                                indices,
                                                tmp);
          for (unsigned int i0 = n2_regular; i0 < (fe_degree - 1) * (fe_degree - 1); ++i0)
            reader.process_dof_gather(indices,
                                      vec,
                                      offset_i2 * (fe_degree - 1) * (fe_degree - 1) + i0,
                                      vec.begin() + offset_i2 * (fe_degree - 1) * (fe_degree - 1) +
                                        i0,
                                      tmp[i0],
                                      std::integral_constant<bool, true>());
          for (unsigned int i1 = 0; i1 < fe_degree - 1; ++i1)
            for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0)
              dof_values[(i1 + 1) * (fe_degree + 1) + i0 + 1] = tmp[i1 * (fe_degree - 1) + i0];
          indices += n_lanes;

          dealii::vectorized_load_and_transpose(
            n_regular, vec.begin() + offset_i2 * (fe_degree - 1) * n_components, indices, tmp);
          for (unsigned int i0 = 0; i0 < n_regular; ++i0)
            dof_values[(fe_degree + 1) * (i0 + 1) + fe_degree] = tmp[i0];
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            reader.process_dof_gather(indices,
                                      vec,
                                      offset_i2 * (fe_degree - 1) + i0,
                                      vec.begin() + offset_i2 * (fe_degree - 1) + i0,
                                      dof_values[(fe_degree + 1) * (i0 + 1) + fe_degree],
                                      std::integral_constant<bool, true>());
          indices += n_lanes;

          // last line
          constexpr unsigned int i = fe_degree * (fe_degree + 1);
          reader.process_dof_gather(indices,
                                    vec,
                                    offset_i2,
                                    vec.begin() + offset_i2,
                                    dof_values[i],
                                    std::integral_constant<bool, true>());
          indices += n_lanes;
          dealii::vectorized_load_and_transpose(n_regular,
                                                vec.begin() +
                                                  offset_i2 * (fe_degree - 1) * n_components,
                                                indices,
                                                dof_values + i + 1);
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            reader.process_dof_gather(indices,
                                      vec,
                                      offset_i2 * (fe_degree - 1) * n_components + i0,
                                      vec.begin() + offset_i2 * (fe_degree - 1) * n_components + i0,
                                      dof_values[i + 1 + i0],
                                      std::integral_constant<bool, true>());
          indices += n_lanes;
          reader.process_dof_gather(indices,
                                    vec,
                                    offset_i2,
                                    vec.begin() + offset_i2,
                                    dof_values[i + fe_degree],
                                    std::integral_constant<bool, true>());
          indices += n_lanes;
        }
      else
        for (unsigned int i1 = 0, i = 0, compressed_i1 = 0, offset_i1 = 0;
             i1 < (dim > 1 ? (fe_degree + 1) : 1);
             ++i1)
          {
            const unsigned int offset =
              (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
            const unsigned int *indices =
              cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
            const unsigned char *unconstrained =
              cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

            // left end point
            if (unconstrained[0])
              for (unsigned int c = 0; c < n_components; ++c)
                reader.process_dof_gather(indices,
                                          vec,
                                          offset * n_components + c,
                                          vec.begin() + offset * n_components + c,
                                          dof_values[i + c * dofs_per_comp],
                                          std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  dof_values[i + c * dofs_per_comp][v] =
                    (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                      0. :
                      vec.local_element(indices[v] + offset * n_components + c);
            ++i;
            indices += n_lanes;

            // interior points of line
            if (unconstrained[1])
              {
                VectorizedArrayType    tmp[fe_degree > 1 ? (fe_degree - 1) * n_components : 1];
                constexpr unsigned int n_regular = (fe_degree - 1) * n_components / 4 * 4;
                dealii::vectorized_load_and_transpose(
                  n_regular, vec.begin() + offset * (fe_degree - 1) * n_components, indices, tmp);
                for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
                  reader.process_dof_gather(indices,
                                            vec,
                                            offset * (fe_degree - 1) * n_components + i0,
                                            vec.begin() + offset * (fe_degree - 1) * n_components +
                                              i0,
                                            tmp[i0],
                                            std::integral_constant<bool, true>());
                for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                  for (unsigned int c = 0; c < n_components; ++c)
                    dof_values[i + c * dofs_per_comp] = tmp[i0 * n_components + c];
              }
            else
              for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    for (unsigned int c = 0; c < n_components; ++c)
                      dof_values[i + c * dofs_per_comp][v] = vec.local_element(
                        indices[v] + (offset * (fe_degree - 1) + i0) * n_components + c);
                  else
                    for (unsigned int c = 0; c < n_components; ++c)
                      dof_values[i + c * dofs_per_comp][v] = 0.;
            indices += n_lanes;

            // right end point
            if (unconstrained[2])
              for (unsigned int c = 0; c < n_components; ++c)
                reader.process_dof_gather(indices,
                                          vec,
                                          offset * n_components + c,
                                          vec.begin() + offset * n_components + c,
                                          dof_values[i + c * dofs_per_comp],
                                          std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  dof_values[i + c * dofs_per_comp][v] =
                    (indices[v] == dealii::numbers::invalid_unsigned_int) ?
                      0. :
                      vec.local_element(indices[v] + offset * n_components + c);
            ++i;

            if (i1 == 0 || i1 == fe_degree - 1)
              {
                ++compressed_i1;
                offset_i1 = 0;
              }
            else
              ++offset_i1;
          }

      if (i2 == 0 || i2 == fe_degree - 1)
        {
          ++compressed_i2;
          offset_i2 = 0;
        }
      else
        ++offset_i2;

      dof_values += n_q_points_1d * n_q_points_1d;
    }
}



template <int dim, int fe_degree, int n_components, typename Number>
void
distribute_local_to_global_compressed(dealii::LinearAlgebra::distributed::Vector<Number> &vec,
                                      const std::vector<unsigned int>  &compressed_indices,
                                      const std::vector<unsigned char> &all_indices_unconstrained,
                                      const unsigned int                cell_no,
                                      dealii::VectorizedArray<Number>  *dof_values)
{
  AssertIndexRange(cell_no * dealii::Utilities::pow(3, dim) *
                     dealii::VectorizedArray<Number>::size(),
                   compressed_indices.size());
  using VectorizedArrayType            = dealii::VectorizedArray<Number>;
  constexpr unsigned int n_q_points_1d = fe_degree + 1;
  constexpr unsigned int n_lanes       = VectorizedArrayType::size();
  const unsigned int    *cell_indices =
    compressed_indices.data() + cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *cell_unconstrained =
    all_indices_unconstrained.data() + cell_no * dealii::Utilities::pow(3, dim);
  constexpr unsigned int dofs_per_comp = dealii::Utilities::pow(n_q_points_1d, dim);
  dealii::internal::VectorDistributorLocalToGlobal<Number, VectorizedArrayType> distributor;

  for (unsigned int i2 = 0, compressed_i2 = 0, offset_i2 = 0; i2 < (dim == 3 ? (fe_degree + 1) : 1);
       ++i2)
    {
      bool all_unconstrained = true;
      for (unsigned int i = 0; i < 9; ++i)
        if (cell_unconstrained[9 * compressed_i2 + i] == 0)
          all_unconstrained = false;
      if (n_components == 1 && fe_degree < 8 && all_unconstrained)
        {
          const unsigned int *indices = cell_indices + 9 * n_lanes * compressed_i2;
          // first line
          distributor.process_dof_gather(indices,
                                         vec,
                                         offset_i2,
                                         vec.begin() + offset_i2,
                                         dof_values[0],
                                         std::integral_constant<bool, true>());
          indices += n_lanes;
          constexpr unsigned int n_regular = (fe_degree - 1) * n_components / 4 * 4;
          dealii::vectorized_transpose_and_store(true,
                                                 n_regular,
                                                 dof_values + 1,
                                                 indices,
                                                 vec.begin() +
                                                   offset_i2 * (fe_degree - 1) * n_components);
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            distributor.process_dof_gather(indices,
                                           vec,
                                           offset_i2 * (fe_degree - 1) * n_components + i0,
                                           vec.begin() +
                                             offset_i2 * (fe_degree - 1) * n_components + i0,
                                           dof_values[1 + i0],
                                           std::integral_constant<bool, true>());
          indices += n_lanes;
          distributor.process_dof_gather(indices,
                                         vec,
                                         offset_i2,
                                         vec.begin() + offset_i2,
                                         dof_values[fe_degree],
                                         std::integral_constant<bool, true>());
          indices += n_lanes;

          // inner part
          VectorizedArrayType tmp[fe_degree > 1 ? (fe_degree - 1) * (fe_degree - 1) : 1];
          for (unsigned int i0 = 0; i0 < n_regular; ++i0)
            tmp[i0] = dof_values[(fe_degree + 1) * (i0 + 1)];
          dealii::vectorized_transpose_and_store(true,
                                                 n_regular,
                                                 tmp,
                                                 indices,
                                                 vec.begin() +
                                                   offset_i2 * (fe_degree - 1) * n_components);
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            distributor.process_dof_gather(indices,
                                           vec,
                                           offset_i2 * (fe_degree - 1) + i0,
                                           vec.begin() + offset_i2 * (fe_degree - 1) + i0,
                                           dof_values[(fe_degree + 1) * (i0 + 1)],
                                           std::integral_constant<bool, true>());
          indices += n_lanes;

          for (unsigned int i1 = 0; i1 < fe_degree - 1; ++i1)
            for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0)
              tmp[i1 * (fe_degree - 1) + i0] = dof_values[(i1 + 1) * (fe_degree + 1) + i0 + 1];
          constexpr unsigned int n2_regular =
            (fe_degree - 1) * (fe_degree - 1) * n_components / 4 * 4;
          dealii::vectorized_transpose_and_store(true,
                                                 n2_regular,
                                                 tmp,
                                                 indices,
                                                 vec.begin() + offset_i2 * (fe_degree - 1) *
                                                                 (fe_degree - 1) * n_components);
          for (unsigned int i0 = n2_regular; i0 < (fe_degree - 1) * (fe_degree - 1); ++i0)
            distributor.process_dof_gather(indices,
                                           vec,
                                           offset_i2 * (fe_degree - 1) * (fe_degree - 1) + i0,
                                           vec.begin() +
                                             offset_i2 * (fe_degree - 1) * (fe_degree - 1) + i0,
                                           tmp[i0],
                                           std::integral_constant<bool, true>());
          indices += n_lanes;

          for (unsigned int i0 = 0; i0 < n_regular; ++i0)
            tmp[i0] = dof_values[(fe_degree + 1) * (i0 + 1) + fe_degree];
          dealii::vectorized_transpose_and_store(true,
                                                 n_regular,
                                                 tmp,
                                                 indices,
                                                 vec.begin() +
                                                   offset_i2 * (fe_degree - 1) * n_components);
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            distributor.process_dof_gather(indices,
                                           vec,
                                           offset_i2 * (fe_degree - 1) + i0,
                                           vec.begin() + offset_i2 * (fe_degree - 1) + i0,
                                           dof_values[(fe_degree + 1) * (i0 + 1) + fe_degree],
                                           std::integral_constant<bool, true>());
          indices += n_lanes;

          // last line
          constexpr unsigned int i = fe_degree * (fe_degree + 1);
          distributor.process_dof_gather(indices,
                                         vec,
                                         offset_i2,
                                         vec.begin() + offset_i2,
                                         dof_values[i],
                                         std::integral_constant<bool, true>());
          indices += n_lanes;
          dealii::vectorized_transpose_and_store(true,
                                                 n_regular,
                                                 dof_values + i + 1,
                                                 indices,
                                                 vec.begin() +
                                                   offset_i2 * (fe_degree - 1) * n_components);
          for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
            distributor.process_dof_gather(indices,
                                           vec,
                                           offset_i2 * (fe_degree - 1) * n_components + i0,
                                           vec.begin() +
                                             offset_i2 * (fe_degree - 1) * n_components + i0,
                                           dof_values[i + 1 + i0],
                                           std::integral_constant<bool, true>());
          indices += n_lanes;
          distributor.process_dof_gather(indices,
                                         vec,
                                         offset_i2,
                                         vec.begin() + offset_i2,
                                         dof_values[i + fe_degree],
                                         std::integral_constant<bool, true>());
          indices += n_lanes;
        }
      else
        for (unsigned int i1 = 0, i = 0, compressed_i1 = 0, offset_i1 = 0;
             i1 < (dim > 1 ? (fe_degree + 1) : 1);
             ++i1)
          {
            const unsigned int offset =
              (compressed_i1 == 1 ? fe_degree - 1 : 1) * offset_i2 + offset_i1;
            const unsigned int *indices =
              cell_indices + 3 * n_lanes * (compressed_i2 * 3 + compressed_i1);
            const unsigned char *unconstrained =
              cell_unconstrained + 3 * (compressed_i2 * 3 + compressed_i1);

            // left end point
            if (unconstrained[0])
              for (unsigned int c = 0; c < n_components; ++c)
                distributor.process_dof_gather(indices,
                                               vec,
                                               offset * n_components + c,
                                               vec.begin() + offset * n_components + c,
                                               dof_values[i + c * dofs_per_comp],
                                               std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v] + offset * n_components + c) +=
                      dof_values[i + c * dofs_per_comp][v];
            ++i;
            indices += n_lanes;

            // interior points of line
            if (unconstrained[1])
              {
                VectorizedArrayType tmp[fe_degree > 1 ? (fe_degree - 1) * n_components : 1];
                for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                  for (unsigned int c = 0; c < n_components; ++c)
                    tmp[i0 * n_components + c] = dof_values[i + c * dofs_per_comp];

                constexpr unsigned int n_regular = (fe_degree - 1) * n_components / 4 * 4;
                dealii::vectorized_transpose_and_store(true,
                                                       n_regular,
                                                       tmp,
                                                       indices,
                                                       vec.begin() +
                                                         offset * (fe_degree - 1) * n_components);
                for (unsigned int i0 = n_regular; i0 < (fe_degree - 1) * n_components; ++i0)
                  distributor.process_dof_gather(indices,
                                                 vec,
                                                 offset * (fe_degree - 1) * n_components + i0,
                                                 vec.begin() +
                                                   offset * (fe_degree - 1) * n_components + i0,
                                                 tmp[i0],
                                                 std::integral_constant<bool, true>());
              }
            else
              for (unsigned int i0 = 0; i0 < fe_degree - 1; ++i0, ++i)
                for (unsigned int v = 0; v < n_lanes; ++v)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    for (unsigned int c = 0; c < n_components; ++c)
                      vec.local_element(indices[v] +
                                        (offset * (fe_degree - 1) + i0) * n_components + c) +=
                        dof_values[i + c * dofs_per_comp][v];
            indices += n_lanes;

            // right end point
            if (unconstrained[2])
              for (unsigned int c = 0; c < n_components; ++c)
                distributor.process_dof_gather(indices,
                                               vec,
                                               offset * n_components + c,
                                               vec.begin() + offset * n_components + c,
                                               dof_values[i + c * dofs_per_comp],
                                               std::integral_constant<bool, true>());
            else
              for (unsigned int v = 0; v < n_lanes; ++v)
                for (unsigned int c = 0; c < n_components; ++c)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v] + offset * n_components + c) +=
                      dof_values[i + c * dofs_per_comp][v];
            ++i;

            if (i1 == 0 || i1 == fe_degree - 1)
              {
                ++compressed_i1;
                offset_i1 = 0;
              }
            else
              ++offset_i1;
          }
      if (i2 == 0 || i2 == fe_degree - 1)
        {
          ++compressed_i2;
          offset_i2 = 0;
        }
      else
        ++offset_i2;
      dof_values += n_q_points_1d * n_q_points_1d;
    }
}


#endif
