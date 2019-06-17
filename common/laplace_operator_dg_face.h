
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include "laplace_operator_dg.h"

namespace MFReference
{
  using namespace dealii;

  template <int dim,
            int fe_degree,
            int n_q_points_1d   = fe_degree + 1,
            typename number     = double,
            typename VectorType = Vector<number>,
            int n_components    = 1>
    class LaplaceOperatorFaceBased
    {
    public:
      LaplaceOperatorFaceBased(const MatrixFree<dim, number> &data,
                               const bool                     zero_within_loop = true,
                               const unsigned int             start_vector_component = 0)
        : data(data)
        , zero_within_loop(zero_within_loop)
        , start_vector_component(start_vector_component)
      {}

      void
      vmult(VectorType &dst, const VectorType &src) const
      {
        if (!zero_within_loop)
          dst = 0;
        data.loop(&LaplaceOperatorFaceBased::local_apply,
                  &LaplaceOperatorFaceBased::local_apply_face,
                  &LaplaceOperatorFaceBased::local_apply_boundary_face,
                  this,
                  dst,
                  src,
                  zero_within_loop,
                  MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                  MatrixFree<dim, number>::DataAccessOnFaces::gradients);
      }

      void
      vmult_add(VectorType &dst, const VectorType &src) const
      {
        data.loop(&LaplaceOperatorFaceBased::local_apply,
                  &LaplaceOperatorFaceBased::local_apply_face,
                  &LaplaceOperatorFaceBased::local_apply_boundary_face,
                  this,
                  dst,
                  src,
                  false,
                  MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                  MatrixFree<dim, number>::DataAccessOnFaces::gradients);
      }

    private:
      void
      local_apply(const MatrixFree<dim, number> &              data,
                  VectorType &                                 dst,
                  const VectorType &                           src,
                  const std::pair<unsigned int, unsigned int> &cell_range) const
      {
        FEEvaluation<dim, fe_degree, n_q_points_1d, n_components, number> phi(
            data, 0, 0, start_vector_component);

        for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
          {
            phi.reinit(cell);
            phi.gather_evaluate(src, false, true);
            for (unsigned int q = 0; q < phi.n_q_points; ++q)
              phi.submit_gradient(phi.get_gradient(q), q);
            phi.integrate_scatter(false, true, dst);
          }
      }

      void
      local_apply_face(
                       const MatrixFree<dim, number> &              data,
                       VectorType &                                 dst,
                       const VectorType &                           src,
                       const std::pair<unsigned int, unsigned int> &face_range) const
      {
        FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components, number>
        fe_eval(data, true, 0, 0, start_vector_component);
        FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components, number>
        fe_eval_neighbor(data, false, 0, 0, start_vector_component);
        typedef typename FEFaceEvaluation<dim,
        fe_degree,
        n_q_points_1d,
        n_components,
        number>::value_type value_type;
        const int actual_degree = data.get_dof_handler().get_fe().degree;

        for (unsigned int face = face_range.first; face < face_range.second; face++)
          {
            fe_eval.reinit(face);
            fe_eval_neighbor.reinit(face);

            fe_eval.gather_evaluate(src, true, true);
            fe_eval_neighbor.gather_evaluate(src, true, true);

            VectorizedArray<number> sigmaF =
              (std::abs((fe_eval.get_normal_vector(0) *
                         fe_eval.inverse_jacobian(0))[dim - 1]) +
               std::abs((fe_eval.get_normal_vector(0) *
                         fe_eval_neighbor.inverse_jacobian(0))[dim - 1])) *
              (number)(actual_degree + 1.0) * (actual_degree + 1.0) * penalty_factor * 0.5;

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                value_type average_value =
                  (fe_eval.get_value(q) - fe_eval_neighbor.get_value(q)) *
                  make_vectorized_array<number>(0.5);
                value_type average_valgrad =
                  fe_eval.get_normal_derivative(q) +
                  fe_eval_neighbor.get_normal_derivative(q);
                average_valgrad =
                  average_value * sigmaF * 2.0 -
                  average_valgrad * make_vectorized_array<number>(0.5);
                fe_eval.submit_normal_derivative(-average_value, q);
                fe_eval_neighbor.submit_normal_derivative(-average_value, q);
                fe_eval.submit_value(average_valgrad, q);
                fe_eval_neighbor.submit_value(-average_valgrad, q);
              }
            fe_eval.integrate_scatter(true, true, dst);
            fe_eval_neighbor.integrate_scatter(true, true, dst);
          }
      }

      void
      local_apply_boundary_face(
                                const MatrixFree<dim, number> &              data,
                                VectorType &                                 dst,
                                const VectorType &                           src,
                                const std::pair<unsigned int, unsigned int> &face_range) const
      {
        FEFaceEvaluation<dim, fe_degree, n_q_points_1d, n_components, number>
        fe_eval(data, true, 0, 0, start_vector_component);
        typedef typename FEFaceEvaluation<dim,
        fe_degree,
        n_q_points_1d,
        n_components,
        number>::value_type value_type;
        const int actual_degree = data.get_dof_handler().get_fe().degree;
        for (unsigned int face = face_range.first; face < face_range.second; face++)
          {
            fe_eval.reinit(face);
            fe_eval.gather_evaluate(src, true, true);
            VectorizedArray<number> sigmaF =
              std::abs((fe_eval.get_normal_vector(0) *
                        fe_eval.inverse_jacobian(0))[dim - 1]) *
              (number)(actual_degree + 1.0) * (actual_degree + 1.0) * penalty_factor * 2.0;

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                value_type average_value   = fe_eval.get_value(q);
                value_type average_valgrad = -fe_eval.get_normal_derivative(q);
                average_valgrad += average_value * sigmaF;
                fe_eval.submit_normal_derivative(-average_value, q);
                fe_eval.submit_value(average_valgrad, q);
              }

            fe_eval.integrate_scatter(true, true, dst);
          }
      }

      const MatrixFree<dim, number> &data;
      const bool                     zero_within_loop;
      const unsigned int             start_vector_component;
    };
}
