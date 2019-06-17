
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/mpi.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "../common/laplace_operator_dg.h"
#include "../common/laplace_operator_dg_face.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace dealii;


constexpr unsigned int min_compiled_degree = 3;
constexpr unsigned int max_compiled_degree = 3;


template <int dim, int degree, int type>
void execute_test(const unsigned int n_cell_steps,
                  const unsigned int n_tests)
{
  ConditionalOStream pcout(std::cout,
                           Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
  std::shared_ptr<FiniteElement<dim>> fe;
  if (type == 0)
    fe.reset(new FE_DGQHermite<dim>(degree));
  else if (type == 1)
    fe.reset(new FE_DGQ<dim>(degree));
  else if (type == 2)
    fe.reset(new FE_DGQArbitraryNodes<dim>(QGauss<1>(degree+1)));
  else
    AssertThrow(false, ExcMessage("Type " + std::to_string(type) + " not implemented"));

  parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD);

  Point<dim> left;
  for (unsigned int d=0; d<dim; ++d)
    left[d] = -1. + 0.05 * (d+1);
  Point<dim> right;
  for (unsigned int d=0; d<dim; ++d)
    right[d] = 0.95 - 0.06 * d;
  std::vector<unsigned int> refinements(dim, 1);
  for (unsigned int c=0; c<n_cell_steps%dim; ++c)
    refinements[c] = 2;
  GridGenerator::subdivided_hyper_rectangle(triangulation, refinements, left, right);

  // deform grid into an affine shape
  Tensor<2,dim> trafo;
  for (unsigned int d=0; d<dim; ++d)
    trafo[d][d] = 1.;
  for (unsigned int d=0; d<dim; ++d)
    for (unsigned int e=0; e<dim; ++e)
      trafo[d][e] += 0.12 * (d+1) * (e+1);
  for (unsigned int i=0; i<triangulation.get_vertices().size(); ++i)
    const_cast<Point<dim>&>(triangulation.get_vertices()[i]) = Point<dim>(trafo * triangulation.get_vertices()[i]);

       triangulation.refine_global (n_cell_steps/dim);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(*fe);
  if (type == 0)
    pcout << "Number of DoFs: " << dof_handler.n_dofs() << std::endl;

  AffineConstraints<double> constraints;
  constraints.close();

  typedef double Number;

  std::shared_ptr<MatrixFree<dim,Number>> matrix_free(new MatrixFree<dim,Number>());
  typename MatrixFree<dim,Number>::AdditionalData mf_data;

  // Do not need face data for the compact scheme in
  // LaplaceOperatorCompactCombine
  mf_data.tasks_parallel_scheme =
    MatrixFree<dim,Number>::AdditionalData::none;
  mf_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                  update_quadrature_points);
  matrix_free->reinit(dof_handler, constraints, QGauss<1>(degree+1), mf_data);

  LinearAlgebra::distributed::Vector<Number> input, output, input_face, reference;

  multigrid::LaplaceOperatorCompactCombine<dim,degree,Number,type> laplace_operator;
  laplace_operator.reinit(matrix_free, 0);
  laplace_operator.initialize_dof_vector(input);
  laplace_operator.initialize_dof_vector(output);
  laplace_operator.initialize_dof_vector(reference);
  for (auto & d : input)
    d = static_cast<Number>(std::rand())/RAND_MAX;

  MatrixFree<dim,Number> matrix_free_ref;
  mf_data.mapping_update_flags_inner_faces = update_gradients | update_JxW_values | update_normal_vectors;
  mf_data.mapping_update_flags_boundary_faces = update_gradients | update_JxW_values | update_normal_vectors;
  matrix_free_ref.reinit(dof_handler, constraints, QGauss<1>(degree+1), mf_data);
  matrix_free_ref.initialize_dof_vector(input_face);
  matrix_free_ref.initialize_dof_vector(reference);

  MFReference::LaplaceOperatorFaceBased<dim,degree,degree+1,Number,
    LinearAlgebra::distributed::Vector<Number>> laplace_operator_ref(matrix_free_ref);
  input_face = input;
  laplace_operator_ref.vmult(reference, input_face);

  Timer time;
  double min_time = 1e10;
  for (unsigned int o=0; o<10; ++o)
    {
      double min = 1e10, avg = 0, max = 0;
      for (unsigned int i=0; i<n_tests; ++i)
        {
          time.restart();
          laplace_operator.vmult(output, input);

          const double t = time.wall_time();
          avg += t;
          min = std::min(t, min);
          max = std::max(t, max);
        }
      avg /= n_tests;

      Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(avg, MPI_COMM_WORLD);
      pcout << "MF mat-vec "
            << std::setw(12) << Utilities::MPI::min(min, MPI_COMM_WORLD) << " "
            << std::setw(12) << data.min << " "
            << std::setw(12) << data.avg << " "
            << std::setw(12) << data.max << " "
            << std::setw(12) << Utilities::MPI::max(max, MPI_COMM_WORLD)
            << std::endl;
      min_time = std::min(min_time, data.max);
    }

  const std::size_t ops_interpolate = (/*add*/2*((degree+1)/2)*2 +
                                       /*mult*/degree+1 +
                                       /*fma*/2*((degree-1)*(degree+1)/2));
  const std::size_t ops_approx = (std::size_t)Utilities::MPI::sum(triangulation.n_locally_owned_active_cells(), MPI_COMM_WORLD)
    *
    (
     4 * dim * ops_interpolate * Utilities::pow(degree+1,dim-1)
     + dim * 2 * dim * Utilities::pow(degree+1,dim)
     +
     (2*dim * (5*(dim-1)*ops_interpolate * Utilities::pow(degree+1,dim-2)
               + (4*dim-1+2+2+3+2*dim)*Utilities::pow(degree+1,dim-1)
               )
      + 4*2* (degree+1 + 2*(degree-1) + 2) * Utilities::pow(degree+1,dim-1)
      + 2*(dim-2)* (2*2) * Utilities::pow(degree+1,dim-1))
     );

  pcout << "Best MF mat-vec " << (type == 0 ? "Hermite" : (type == 1 ? "DGQ_GL " : "DGQ_G  "))
        << " n_dof= " << std::setw(12) << std::left << dof_handler.n_dofs()
        << std::setw(12) << min_time  << "   DoFs/s "
        << dof_handler.n_dofs()/min_time
        << "    GFlop/s "
        << 1e-9*ops_approx / min_time
        << "    GB/s "
        << 1e-9*dof_handler.n_dofs()*sizeof(Number)*3/min_time
        << std::endl;
  output -= reference;
  pcout << "Verification of result: " << output.linfty_norm() << std::endl
        << std::endl;
}

template <int dim, int degree>
void run_test(const unsigned int given_degree,
              const int n_cell_steps,
              const unsigned int n_tests)
{
  AssertThrow(given_degree >= min_compiled_degree &&
              given_degree <= max_compiled_degree,
              ExcNotImplemented("degree " + std::to_string(given_degree)
                                + " not implemented"));
  if (given_degree > degree)
    run_test<dim,(degree > max_compiled_degree ? degree : degree+1)>
      (given_degree, n_cell_steps, n_tests);
  else
    for (unsigned int cycle = 0; cycle < (n_cell_steps < 0 ? 40 : 1); ++cycle)
      {
        const unsigned int my_cycle = n_cell_steps >= 0 ? n_cell_steps : cycle;
        execute_test<dim,degree,0>(my_cycle, n_tests);
        execute_test<dim,degree,1>(my_cycle, n_tests);
        execute_test<dim,degree,2>(my_cycle, n_tests);
      }
}


int main(int argc, char** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

  const unsigned int dim = 2;
  unsigned int degree = 3;
  int n_refinement_steps = -1;
  unsigned int nsteps = 1000;
  if (argc > 1)
    degree = atoi(argv[1]);
  if (argc > 2)
    n_refinement_steps = atoi(argv[2]);
  if (argc > 3)
    nsteps = atoi(argv[3]);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::cout << "Number of MPI processes:        "
                << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)
                << std::endl;
      unsigned int nthreads = 1;
#ifdef _OPENMP
      nthreads = omp_get_max_threads();
      std::cout << "Number of OpenMP threads:       " << nthreads << std::endl;
#endif
      std::cout << "Degree of element:              " << degree << std::endl;
      std::cout << std::endl;
    }

  run_test<dim,min_compiled_degree>(degree, n_refinement_steps, nsteps);
}
