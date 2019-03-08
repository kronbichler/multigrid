
#ifndef renumber_dofs_for_mf_h
#define renumber_dofs_for_mf_h

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/matrix_free/matrix_free.h>


template <int dim, typename Number>
void renumber_dofs_mf(dealii::DoFHandler<dim> &dof_handler,
                      const dealii::AffineConstraints<double> &constraints,
                      const typename dealii::MatrixFree<dim,Number>::AdditionalData &mf_data)
{
  typename dealii::MatrixFree<dim,Number>::AdditionalData my_mf_data =
    mf_data;
  my_mf_data.initialize_mapping = false;
  dealii::MatrixFree<dim,Number> matrix_free;
  matrix_free.reinit(dof_handler, constraints,
                     dealii::QGauss<1>(dof_handler.get_fe().degree+1),
                     my_mf_data);

  const dealii::IndexSet &index_set = mf_data.level_mg_handler==dealii::numbers::invalid_unsigned_int ?
    dof_handler.locally_owned_dofs() : dof_handler.locally_owned_mg_dofs(mf_data.level_mg_handler);
  std::vector<std::vector<unsigned int>> processors_involved(index_set.n_elements());
  std::vector<dealii::types::global_dof_index> dof_indices(dof_handler.get_fe().dofs_per_cell);
  if (mf_data.level_mg_handler == dealii::numbers::invalid_unsigned_int)
    {
      for (auto &cell : dof_handler.active_cell_iterators())
        if (cell->is_ghost())
          {
            cell->get_dof_indices(dof_indices);
            for (auto i : dof_indices)
              {
                if (dof_handler.locally_owned_dofs().is_element(i))
                  processors_involved[dof_handler.locally_owned_dofs().index_within_set(i)]
                    .push_back(cell->subdomain_id());
              }
          }
    }
  else
    {
      for (auto &cell : dof_handler.mg_cell_iterators())
        if (cell->level_subdomain_id() != dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) &&
            cell->level_subdomain_id() < dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) )
          {
            cell->get_active_or_mg_dof_indices(dof_indices);
            for (auto i : dof_indices)
              {
                if (index_set.is_element(i))
                  processors_involved[index_set.index_within_set(i)]
                    .push_back(cell->level_subdomain_id());
              }
          }
    }
  // sort & compress out duplicates
  for (std::vector<unsigned int> &v : processors_involved)
    if (v.size() > 1)
      {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
      }
  std::map<std::vector<unsigned int>, std::vector<unsigned int>,
           std::function<bool(const std::vector<unsigned int> &,
                              const std::vector<unsigned int> &)>>
    sorted_entries{[](const std::vector<unsigned int> &a,
                      const std::vector<unsigned int> &b)
      {
        if (a.size() < b.size())
          return true;
        if (a.size() == b.size())
          {
            for (unsigned int i=0; i<a.size(); ++i)
              if (a[i] < b[i])
                return true;
              else if (a[i] > b[i])
                return false;
          }
        return false;
      }
  };
  for (unsigned int i=0; i<processors_involved.size(); ++i)
    {
      std::vector<unsigned int> &v = sorted_entries[processors_involved[i]];
      v.push_back(i);
    }

  std::vector<unsigned int> numbers_mf_order(index_set.n_elements(),
                                             dealii::numbers::invalid_unsigned_int);
  unsigned int counter_dof_numbers = 0;
  std::vector<unsigned char> touch_count(index_set.n_elements());
  std::vector<unsigned int> local_dofs_mf;
  const unsigned int fe_degree = dof_handler.get_fe().degree;
  for (unsigned int cell=0; cell<matrix_free.n_cell_batches(); ++cell)
    {
      matrix_free.get_dof_info().get_dof_indices_on_cell_batch(local_dofs_mf, cell, true);
      std::sort(local_dofs_mf.begin(), local_dofs_mf.end());
      local_dofs_mf.erase(std::unique(local_dofs_mf.begin(), local_dofs_mf.end()),
                          local_dofs_mf.end());
      for (unsigned int i : local_dofs_mf)
        if (i<index_set.n_elements())
          touch_count[i]++;

      // inject the natural ordering of the matrix-free loop order
      for (unsigned int v=0; v<matrix_free.n_components_filled(cell); ++v)
        {
          if (mf_data.level_mg_handler == dealii::numbers::invalid_unsigned_int)
            matrix_free.get_cell_iterator(cell, v)->get_dof_indices(dof_indices);
          else
            typename dealii::DoFHandler<dim>::level_cell_iterator(matrix_free.get_cell_iterator(cell, v))->get_active_or_mg_dof_indices(dof_indices);
          unsigned int cf=0;
          for ( ; cf<dealii::GeometryInfo<dim>::vertices_per_cell; ++cf)
            if (index_set.is_element(dof_indices[cf]) &&
                numbers_mf_order[index_set.index_within_set(dof_indices[cf])] ==
                dealii::numbers::invalid_unsigned_int)
              numbers_mf_order[index_set.
                               index_within_set(dof_indices[cf])] = counter_dof_numbers++;
          for (unsigned int line=0; line<dealii::GeometryInfo<dim>::lines_per_cell; ++line)
            {
              if (index_set.is_element(dof_indices[cf]) &&
                  numbers_mf_order[index_set.index_within_set(dof_indices[cf])] ==
                  dealii::numbers::invalid_unsigned_int)
                for (unsigned int i=0; i<fe_degree-1; ++i, ++cf)
                  numbers_mf_order[index_set.
                                   index_within_set(dof_indices[cf])] = counter_dof_numbers++;
              else
                cf += fe_degree-1;
            }
          for (unsigned int quad=0; quad<dealii::GeometryInfo<dim>::quads_per_cell; ++quad)
            {
              if (index_set.is_element(dof_indices[cf]) &&
                  numbers_mf_order[index_set.index_within_set(dof_indices[cf])] ==
                  dealii::numbers::invalid_unsigned_int)
                for (unsigned int i=0; i<(fe_degree-1)*(fe_degree-1); ++i, ++cf)
                  numbers_mf_order[index_set.
                                   index_within_set(dof_indices[cf])] = counter_dof_numbers++;
              else
                cf += (fe_degree-1)*(fe_degree-1);
            }
          for (unsigned int hex=0; hex<dealii::GeometryInfo<dim>::hexes_per_cell; ++hex)
            {
              if (index_set.is_element(dof_indices[cf]) &&
                  numbers_mf_order[index_set.
                                   index_within_set(dof_indices[cf])] ==
                  dealii::numbers::invalid_unsigned_int)
                for (unsigned int i=0; i<(fe_degree-1)*(fe_degree-1)*(fe_degree-1); ++i, ++cf)
                  numbers_mf_order[index_set.
                                   index_within_set(dof_indices[cf])] = counter_dof_numbers++;
              else
                cf += (fe_degree-1)*(fe_degree-1)*(fe_degree-1);
            }
        }
    }
  AssertThrow(counter_dof_numbers == index_set.n_elements(),
              dealii::ExcMessage
              ("Expected " + std::to_string(index_set.n_elements())
               + " dofs, touched " + std::to_string(counter_dof_numbers)));


  std::vector<unsigned int> new_numbers;
  new_numbers.reserve(index_set.n_elements());
  std::vector<unsigned int> single_domain_dofs = sorted_entries[std::vector<unsigned int>()];
  sorted_entries.erase(std::vector<unsigned int>());
  for (auto i : single_domain_dofs)
    if (touch_count[i] == 1)
      new_numbers.push_back(i);

  // renumber within the chunks to reflect the numbering we want to impose
  // through the matrix-free loop
  auto comp = [&](const unsigned int a, const unsigned int b) {
    return (numbers_mf_order[a] < numbers_mf_order[b]);
  };
  std::sort(new_numbers.begin(), new_numbers.end(), comp);
  const unsigned int single_size = new_numbers.size();

  for (auto i : single_domain_dofs)
    if (touch_count[i] > 1)
      new_numbers.push_back(i);
  std::sort(new_numbers.begin()+single_size, new_numbers.end(), comp);
  const unsigned int multiple_size = new_numbers.size() - single_size;

  for (auto &it : sorted_entries)
    for (auto i : it.second)
      new_numbers.push_back(i);
  std::sort(new_numbers.begin()+single_size+multiple_size, new_numbers.end(), comp);
  const unsigned int multiproc_size = new_numbers.size() - single_size - multiple_size;

  for (auto i : single_domain_dofs)
    if (touch_count[i] == 0)
      new_numbers.push_back(i);
  std::sort(new_numbers.begin()+single_size+multiple_size+multiproc_size, new_numbers.end(), comp);

  //std::cout << "P" << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " "
  //          << single_size << " " << multiple_size << " " << multiproc_size << std::endl;

  AssertThrow(new_numbers.size() == index_set.n_elements(),
              dealii::ExcMessage("Dimension mismatch " + std::to_string(new_numbers.size()) +
                                 " vs " + std::to_string(index_set.n_elements())));

  std::vector<dealii::types::global_dof_index> new_global_numbers
    (index_set.n_elements());
  for (unsigned int i=0; i<new_numbers.size(); ++i)
    new_global_numbers[new_numbers[i]] = index_set.nth_index_in_set(i);

  //std::cout << "Numbers on p" << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << " ";
  //for (auto i : new_global_numbers)
  //  std::cout << i << " ";
  //std::cout << std::endl;

  if (mf_data.level_mg_handler == dealii::numbers::invalid_unsigned_int)
    dof_handler.renumber_dofs(new_global_numbers);
  else
    dof_handler.renumber_dofs(mf_data.level_mg_handler, new_global_numbers);

  if (false)
    {
      dealii::IndexSet locally_active_dofs;
      dealii::DoFTools::extract_locally_active_dofs(dof_handler, locally_active_dofs);
      dealii::Utilities::MPI::Partitioner partitioner;
      partitioner.reinit(dof_handler.locally_owned_dofs(), locally_active_dofs,
                         MPI_COMM_WORLD);
      std::cout << dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
                << " n_import_indices " << partitioner.n_import_indices()
                << " import_indices.size() " << partitioner.import_indices().size()
                << std::endl;
    }
}

#endif
