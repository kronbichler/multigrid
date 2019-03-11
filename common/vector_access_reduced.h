
#ifndef vector_access_reduced_h
#define vector_access_reduced_h

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>

template <int dim, int fe_degree, typename Number>
void read_dof_values_compressed(const dealii::LinearAlgebra::distributed::Vector<Number> &vec,
                                const std::vector<unsigned int> &compressed_indices,
                                const std::vector<unsigned char> &all_indices_unconstrained,
                                const unsigned int cell_no,
                                dealii::VectorizedArray<Number> *dof_values)
{
  AssertIndexRange(cell_no*dealii::Utilities::pow(3, dim) *
                   dealii::VectorizedArray<Number>::n_array_elements,
                   compressed_indices.size());
  constexpr unsigned int n_lanes = dealii::VectorizedArray<Number>::n_array_elements;
  const unsigned int *indices = compressed_indices.data() +
    cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *unconstrained = all_indices_unconstrained.data() +
    cell_no * dealii::Utilities::pow(3, dim);
  dealii::internal::VectorReader<Number> reader;
  // vertex dofs
  for (unsigned int i2=0; i2<(dim==3?2:1); ++i2)
    for (unsigned int i1=0; i1<2; ++i1)
      for (unsigned int i0=0; i0<2; ++i0, indices+=n_lanes, ++unconstrained)
        if (*unconstrained)
          reader.process_dof_gather(indices, vec, 0,
                                    dof_values[i2*fe_degree*(fe_degree+1)*(fe_degree+1)+
                                               i1*fe_degree*(fe_degree+1) +
                                               i0*fe_degree],
                                    std::integral_constant<bool, true>());
        else
          for (unsigned int v=0; v<n_lanes; ++v)
            dof_values[i2*fe_degree*(fe_degree+1)*(fe_degree+1)+
                       i1*fe_degree*(fe_degree+1) +
                       i0*fe_degree][v] = (indices[v] == dealii::numbers::invalid_unsigned_int) ?
              0. : vec.local_element(indices[v]);

  // line dofs
  constexpr unsigned int offsets[2][4] =
    {{fe_degree+1, 2*fe_degree+1, 1, (fe_degree+1)*fe_degree+1},
     {fe_degree*(fe_degree+1)*(fe_degree+1)+fe_degree+1,
      fe_degree*(fe_degree+1)*(fe_degree+1)+2*fe_degree+1,
      fe_degree*(fe_degree+1)*(fe_degree+1)+1,
      (fe_degree+1)*(fe_degree+1)*(fe_degree+1)-fe_degree}};
  for (unsigned int d=0; d<dim-1; ++d)
    {
      for (unsigned int l=0; l<2; ++l)
        {
          if (*unconstrained)
            for (unsigned int i=0; i<fe_degree-1; ++i)
              reader.process_dof_gather(indices, vec, i,
                                        dof_values[offsets[d][l]+i*(fe_degree+1)],
                                        std::integral_constant<bool, true>());
          else
            for (unsigned int i=0; i<fe_degree-1; ++i)
              for (unsigned int v=0; v<n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  dof_values[offsets[d][l]+i*(fe_degree+1)][v] = vec.local_element(indices[v]+i);
                else
                  dof_values[offsets[d][l]+i*(fe_degree+1)][v] = 0.;
          indices += n_lanes;
          ++unconstrained;
        }
      for (unsigned int l=2; l<4; ++l)
        {
          if (*unconstrained)
            {
              constexpr unsigned int n_regular = (fe_degree-1)/4*4;
              dealii::vectorized_load_and_transpose(n_regular,
                                                    vec.begin(), indices,
                                                    dof_values + offsets[d][l]);
              for (unsigned int i=n_regular; i<fe_degree-1; ++i)
                reader.process_dof_gather(indices, vec, i,
                                          dof_values[offsets[d][l]+i],
                                          std::integral_constant<bool, true>());

            }
          else
            for (unsigned int i=0; i<fe_degree-1; ++i)
              for (unsigned int v=0; v<n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  dof_values[offsets[d][l]+i][v] = vec.local_element(indices[v]+i);
                else
                  dof_values[offsets[d][l]+i][v] = 0;;
          indices += n_lanes;
          ++unconstrained;
        }
    }
  if (dim==3)
    {
      constexpr unsigned int strides2 = (fe_degree+1)*(fe_degree+1);
      constexpr unsigned int offsets2[4] = {strides2, strides2+fe_degree, strides2+(fe_degree+1)*fe_degree, strides2+(fe_degree+1)*(fe_degree+1)-1};
      for (unsigned int l=0; l<4; ++l)
        {
          if (*unconstrained)
            for (unsigned int i=0; i<fe_degree-1; ++i)
              reader.process_dof_gather(indices, vec, i,
                                        dof_values[offsets2[l]+i*strides2],
                                        std::integral_constant<bool, true>());
          else
            for (unsigned int i=0; i<fe_degree-1; ++i)
              for (unsigned int v=0; v<n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  dof_values[offsets2[l]+i*strides2][v] = vec.local_element(indices[v]+i);
                else
                  dof_values[offsets2[l]+i*strides2][v] = 0.;
          indices += n_lanes;
          ++unconstrained;
        }

      // face dofs
      for (unsigned int face=0; face<6; ++face)
        {
          const unsigned int stride1 = dealii::Utilities::pow(fe_degree+1,(face/2+1)%dim);
          const unsigned int stride2 = dealii::Utilities::pow(fe_degree+1,(face/2+2)%dim);
          const unsigned int offset  = ((face%2==0) ? 0 : fe_degree) * dealii::Utilities::pow(fe_degree+1,face/2);
          if (*unconstrained)
            for (unsigned int i2=1, j=0; i2<fe_degree; ++i2)
              for (unsigned int i1=1; i1<fe_degree; ++i1, ++j)
                reader.process_dof_gather(indices, vec, j,
                                          dof_values[offset + i2*stride2 + i1*stride1],
                                          std::integral_constant<bool, true>());
          else
            for (unsigned int i2=1, j=0; i2<fe_degree; ++i2)
              for (unsigned int i1=1; i1<fe_degree; ++i1, ++j)
                for (unsigned int v=0; v<n_lanes; ++v)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    dof_values[offset + i2*stride2 + i1*stride1][v] = vec.local_element(indices[v]+j);
                  else
                    dof_values[offset + i2*stride2 + i1*stride1][v] = 0;
          indices += n_lanes;
          ++unconstrained;
        }
    }

  // cell dofs
  if (*unconstrained)
    for (unsigned int i2=1, j=0; i2<(dim==3?fe_degree:2); ++i2)
      for (unsigned int i1=1; i1<fe_degree; ++i1)
        {
          constexpr unsigned int n_regular = (fe_degree-1)/4*4;
          dealii::vectorized_load_and_transpose(n_regular,
                                                vec.begin()+j,
                                                indices,
                                                dof_values + i2*(fe_degree+1)*(fe_degree+1)+i1*(fe_degree+1)+1);
          j+=n_regular;
          for (unsigned int i0=1+n_regular; i0<fe_degree; ++i0, ++j)
            reader.process_dof_gather(indices, vec, j,
                                      dof_values[i2*(fe_degree+1)*(fe_degree+1)+
                                                 i1*(fe_degree+1)+i0],
                                      std::integral_constant<bool, true>());
        }
  else
    for (unsigned int i2=1, j=0; i2<(dim==3?fe_degree:2); ++i2)
      for (unsigned int i1=1; i1<fe_degree; ++i1)
        for (unsigned int i0=1; i0<fe_degree; ++i0, ++j)
          for (unsigned int v=0; v<n_lanes; ++v)
            if (indices[v] != dealii::numbers::invalid_unsigned_int)
              dof_values[i2*(fe_degree+1)*(fe_degree+1)+i1*(fe_degree+1)+i0][v] = vec.local_element(indices[v]+j);
            else
              dof_values[i2*(fe_degree+1)*(fe_degree+1)+i1*(fe_degree+1)+i0][v] = 0;
  /*
  constexpr unsigned int strides[4] = {fe_degree+1, fe_degree+1, 1, 1};
  constexpr unsigned int offsets[4] = {0, fe_degree, 0, (fe_degree+1)*fe_degree};
  for (unsigned int l=0; l<4; ++l)
    {
      for (unsigned int i=1; i<fe_degree; ++i)
        for (unsigned int v=0; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
          if (indices[v] == dealii::numbers::invalid_unsigned_int)
            dof_values[offsets[l]+i*strides[l]][v] = 0.;
          else
            dof_values[offsets[l]+i*strides[l]][v] = vec.local_element(indices[v]+i-1);
      indices += dealii::VectorizedArray<Number>::n_array_elements;
    }
  if (dim==3)
    {
      for (unsigned int l=0; l<4; ++l)
        {
          for (unsigned int i=1; i<fe_degree; ++i)
            for (unsigned int v=0; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
              if (indices[v] == dealii::numbers::invalid_unsigned_int)
                dof_values[fe_degree*(fe_degree+1)*(fe_degree+1)+
                           offsets[l]+i*strides[l]][v] = 0.;
              else
                dof_values[fe_degree*(fe_degree+1)*(fe_degree+1)+
                           offsets[l]+i*strides[l]][v] = vec.local_element(indices[v]+i-1);
          indices += dealii::VectorizedArray<Number>::n_array_elements;
        }
      constexpr unsigned int strides2 = (fe_degree+1)*(fe_degree+1);
      constexpr unsigned int offsets2[4] = {0, fe_degree, (fe_degree+1)*fe_degree, (fe_degree+1)*(fe_degree+1)-1};
      for (unsigned int l=0; l<4; ++l)
        {
          for (unsigned int i=1; i<fe_degree; ++i)
            for (unsigned int v=0; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
              if (indices[v] == dealii::numbers::invalid_unsigned_int)
                dof_values[offsets2[l]+i*strides2][v] = 0.;
              else
                for (unsigned int i=1; i<fe_degree; ++i)
                  dof_values[offsets2[l]+i*strides2][v] = vec.local_element(indices[v]+i-1);
          indices += dealii::VectorizedArray<Number>::n_array_elements;
        }

      // face dofs
      for (unsigned int face=0; face<6; ++face)
        {
          const unsigned int stride1 = dealii::Utilities::pow(fe_degree+1,(face/2+1)%dim);
          const unsigned int stride2 = dealii::Utilities::pow(fe_degree+1,(face/2+2)%dim);
          const unsigned int offset  = ((face%2==0) ? 0 : fe_degree) * dealii::Utilities::pow(fe_degree+1,face/2);
          for (unsigned int i2=1, j=0; i2<fe_degree; ++i2)
            for (unsigned int i1=1; i1<fe_degree; ++i1, ++j)
              for (unsigned int v=0; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
                if (indices[v] == dealii::numbers::invalid_unsigned_int)
                  dof_values[offset + i2*stride2 + i1*stride1][v] = 0.;
                else
                  dof_values[offset + i2*stride2 + i1*stride1][v] = vec.local_element(indices[v]+j);
          indices += dealii::VectorizedArray<Number>::n_array_elements;
        }
    }

  // cell dofs
  for (unsigned int i2=1, j=0; i2<(dim==3?fe_degree:2); ++i2)
    for (unsigned int i1=1; i1<fe_degree; ++i1)
      for (unsigned int i0=1; i0<fe_degree; ++i0, ++j)
        for (unsigned int v=0; v<dealii::VectorizedArray<Number>::n_array_elements; ++v)
          if (indices[v] == dealii::numbers::invalid_unsigned_int)
            dof_values[i2*(fe_degree+1)*(fe_degree+1)+i1*(fe_degree+1)+i0][v] = 0.;
          else
            dof_values[i2*(fe_degree+1)*(fe_degree+1)+i1*(fe_degree+1)+i0][v] =
              vec.local_element(indices[v]+j);
  */
}



template <int dim, int fe_degree, typename Number>
void distribute_local_to_global_compressed
(dealii::LinearAlgebra::distributed::Vector<Number> &vec,
 const std::vector<unsigned int> &compressed_indices,
 const std::vector<unsigned char> &all_indices_unconstrained,
 const unsigned int cell_no,
 dealii::VectorizedArray<Number> *dof_values)
{
  AssertIndexRange(cell_no*dealii::Utilities::pow(3, dim) *
                   dealii::VectorizedArray<Number>::n_array_elements,
                   compressed_indices.size());
  constexpr unsigned int n_lanes = dealii::VectorizedArray<Number>::n_array_elements;
  const unsigned int *indices = compressed_indices.data() +
    cell_no * n_lanes * dealii::Utilities::pow(3, dim);
  const unsigned char *unconstrained = all_indices_unconstrained.data() +
    cell_no * dealii::Utilities::pow(3, dim);
  dealii::internal::VectorDistributorLocalToGlobal<Number> distributor;

  // vertex dofs
  for (unsigned int i2=0; i2<(dim==3?2:1); ++i2)
    for (unsigned int i1=0; i1<2; ++i1)
      for (unsigned int i0=0; i0<2; ++i0, indices+=n_lanes, ++unconstrained)
        if (*unconstrained)
          distributor.process_dof_gather(indices, vec, 0,
                                         dof_values[i2*fe_degree*(fe_degree+1)*(fe_degree+1)+
                                                    i1*fe_degree*(fe_degree+1) +
                                                    i0*fe_degree],
                                         std::integral_constant<bool, true>());
        else
          for (unsigned int v=0; v<n_lanes; ++v)
            if (indices[v] != dealii::numbers::invalid_unsigned_int)
              vec.local_element(indices[v]) += dof_values[i2*fe_degree*(fe_degree+1)*(fe_degree+1)+
                                                          i1*fe_degree*(fe_degree+1) +
                                                          i0*fe_degree][v];

  // line dofs
  constexpr unsigned int offsets[2][4] =
    {{fe_degree+1, 2*fe_degree+1, 1, (fe_degree+1)*fe_degree+1},
     {fe_degree*(fe_degree+1)*(fe_degree+1)+fe_degree+1,
      fe_degree*(fe_degree+1)*(fe_degree+1)+2*fe_degree+1,
      fe_degree*(fe_degree+1)*(fe_degree+1)+1,
      (fe_degree+1)*(fe_degree+1)*(fe_degree+1)-fe_degree}};
  for (unsigned int d=0; d<dim-1; ++d)
    {
      for (unsigned int l=0; l<2; ++l)
        {
          if (*unconstrained)
            for (unsigned int i=0; i<fe_degree-1; ++i)
              distributor.process_dof_gather(indices, vec, i,
                                             dof_values[offsets[d][l]+i*(fe_degree+1)],
                                             std::integral_constant<bool, true>());
          else
            for (unsigned int i=0; i<fe_degree-1; ++i)
              for (unsigned int v=0; v<n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  vec.local_element(indices[v]+i) += dof_values[offsets[d][l]+i*(fe_degree+1)][v];
          indices += n_lanes;
          ++unconstrained;
        }
      for (unsigned int l=2; l<4; ++l)
        {
          if (*unconstrained)
            {
              constexpr unsigned int n_regular = (fe_degree-1)/4*4;
              dealii::vectorized_transpose_and_store(true, n_regular,
                                                     dof_values + offsets[d][l], indices,
                                                     vec.begin());
              for (unsigned int i=n_regular; i<fe_degree-1; ++i)
                distributor.process_dof_gather(indices, vec, i,
                                               dof_values[offsets[d][l]+i],
                                               std::integral_constant<bool, true>());

            }
          else
            for (unsigned int i=0; i<fe_degree-1; ++i)
              for (unsigned int v=0; v<n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  vec.local_element(indices[v]+i) += dof_values[offsets[d][l]+i][v];
          indices += n_lanes;
          ++unconstrained;
        }
    }
  if (dim==3)
    {
      constexpr unsigned int strides2 = (fe_degree+1)*(fe_degree+1);
      constexpr unsigned int offsets2[4] = {strides2, strides2+fe_degree, strides2+(fe_degree+1)*fe_degree, strides2+(fe_degree+1)*(fe_degree+1)-1};
      for (unsigned int l=0; l<4; ++l)
        {
          if (*unconstrained)
            for (unsigned int i=0; i<fe_degree-1; ++i)
              distributor.process_dof_gather(indices, vec, i,
                                             dof_values[offsets2[l]+i*strides2],
                                             std::integral_constant<bool, true>());
          else
            for (unsigned int i=0; i<fe_degree-1; ++i)
              for (unsigned int v=0; v<n_lanes; ++v)
                if (indices[v] != dealii::numbers::invalid_unsigned_int)
                  vec.local_element(indices[v]+i) += dof_values[offsets2[l]+i*strides2][v];
          indices += n_lanes;
          ++unconstrained;
        }

      // face dofs
      for (unsigned int face=0; face<6; ++face)
        {
          const unsigned int stride1 = dealii::Utilities::pow(fe_degree+1,(face/2+1)%dim);
          const unsigned int stride2 = dealii::Utilities::pow(fe_degree+1,(face/2+2)%dim);
          const unsigned int offset  = ((face%2==0) ? 0 : fe_degree) * dealii::Utilities::pow(fe_degree+1,face/2);
          if (*unconstrained)
            for (unsigned int i2=1, j=0; i2<fe_degree; ++i2)
              for (unsigned int i1=1; i1<fe_degree; ++i1, ++j)
                distributor.process_dof_gather(indices, vec, j,
                                               dof_values[offset + i2*stride2 + i1*stride1],
                                               std::integral_constant<bool, true>());
          else
            for (unsigned int i2=1, j=0; i2<fe_degree; ++i2)
              for (unsigned int i1=1; i1<fe_degree; ++i1, ++j)
                for (unsigned int v=0; v<n_lanes; ++v)
                  if (indices[v] != dealii::numbers::invalid_unsigned_int)
                    vec.local_element(indices[v]+j) += dof_values[offset + i2*stride2 + i1*stride1][v];
          indices += n_lanes;
          ++unconstrained;
        }
    }

  // cell dofs
  if (*unconstrained)
    for (unsigned int i2=1, j=0; i2<(dim==3?fe_degree:2); ++i2)
      for (unsigned int i1=1; i1<fe_degree; ++i1)
        {
          constexpr unsigned int n_regular = (fe_degree-1)/4*4;
          dealii::vectorized_transpose_and_store(true, n_regular,
                                                 dof_values + i2*(fe_degree+1)*(fe_degree+1)+i1*(fe_degree+1)+1,
                                                 indices,
                                                 vec.begin()+j);
          j+=n_regular;
          for (unsigned int i0=1+n_regular; i0<fe_degree; ++i0, ++j)
            distributor.process_dof_gather(indices, vec, j,
                                           dof_values[i2*(fe_degree+1)*(fe_degree+1)+
                                                      i1*(fe_degree+1)+i0],
                                           std::integral_constant<bool, true>());
        }
  else
    for (unsigned int i2=1, j=0; i2<(dim==3?fe_degree:2); ++i2)
      for (unsigned int i1=1; i1<fe_degree; ++i1)
        for (unsigned int i0=1; i0<fe_degree; ++i0, ++j)
          for (unsigned int v=0; v<n_lanes; ++v)
            if (indices[v] != dealii::numbers::invalid_unsigned_int)
              vec.local_element(indices[v]+j) +=
                dof_values[i2*(fe_degree+1)*(fe_degree+1)+i1*(fe_degree+1)+i0][v];
}


#endif
