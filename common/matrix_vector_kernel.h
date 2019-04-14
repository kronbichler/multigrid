// This file is free software. You can use it, redistribute it, and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 2.1 of the License, or (at
// your option) any later version.
//
// This file is inspired by the file
// deal.II/include/deal.II/matrix_free/tensor_product_kernels.h, see
// www.dealii.org for information about licenses. Here is the original deal.II
// license statement:
// ---------------------------------------------------------------------
//
// Copyright (C) 2017 - 2018 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


#ifndef matrix_vector_kernel_h
#define matrix_vector_kernel_h


template <int nn, int stride, int type, bool contract_over_rows, bool add_into,
          typename Number, typename Number2=Number, bool nontemporal_store=false, int do_dg = 0>
inline ALWAYS_INLINE
void
apply_1d_matvec_kernel(const Number2 *__restrict coefficients_eo,
                       const Number* in,
                       Number* out,
                       const Number* array_for_add = nullptr,
                       const Number2* __restrict dg_coefficients = nullptr,
                       Number* array_face = nullptr)
{
  const unsigned int mid = nn/2;
  const unsigned int offset = (nn+1)/2;
  Number xp[mid>0?mid:1], xm[mid>0?mid:1];
  for (unsigned int i=0; i<mid; ++i)
    {
      if (contract_over_rows == true && type == 1)
        {
          xp[i] = in[stride*i] - in[stride*(nn-1-i)];
          xm[i] = in[stride*i] + in[stride*(nn-1-i)];
        }
      else
        {
          xp[i] = in[i*stride] + in[(nn-1-i)*stride];
          xm[i] = in[i*stride] - in[(nn-1-i)*stride];
        }
    }

  static_assert(do_dg == 0 || type==1, "Not implemented");

  Number xmid = in[stride*mid];
  if (contract_over_rows == true)
    {
      for (unsigned int col=0; col<mid; ++col)
        {
          Number r0, r1;
          r0 = coefficients_eo[col]                 * xp[0];
          r1 = coefficients_eo[(nn-1)*offset + col] * xm[0];

          for (unsigned int ind=1; ind<mid; ++ind)
            {
              r0 += coefficients_eo[ind*offset+col]        * xp[ind];
              r1 += coefficients_eo[(nn-1-ind)*offset+col] * xm[ind];
            }
          if (nn % 2 == 1)
            {
              if (type == 1)
                r1 += coefficients_eo[mid*offset+col] * xmid;
              else
                r0 += coefficients_eo[mid*offset+col] * xmid;
            }

          Number t = r0;
          r0 = (add_into ? array_for_add[col*stride] + t : t) + r1;
          r1 = (add_into ? array_for_add[(nn-1-col)*stride] + t : t) - r1;
          if (nontemporal_store == false)
            {
              out[col*stride]        = r0;
              out[(nn-1-col)*stride] = r1;
            }
          else
            {
              r0.streaming_store(&out[col*stride][0]);
              r1.streaming_store(&out[(nn-1-col)*stride][0]);
            }
        }
      if (nn % 2 == 1)
        {
          Number r0 = (add_into ?
                       array_for_add[mid] + coefficients_eo[mid*offset+mid] * xmid :
                       coefficients_eo[mid*offset+mid]*xmid);
          for (unsigned int ind=0; ind<mid; ++ind)
            r0 += coefficients_eo[ind*offset+mid] * xp[ind];

          if (nontemporal_store == false)
            out[mid*stride] = r0;
          else
            r0.streaming_store(&out[mid*stride][0]);
        }
      if (do_dg > 0)
        {
          Number r0 = dg_coefficients[0] * xm[0];
          Number r1 = dg_coefficients[nn-1] * xp[0];
          for (unsigned int ind=1; ind<mid; ++ind)
            {
              r0 += dg_coefficients[ind] * xm[ind];
              r1 += dg_coefficients[nn-1-ind] * xp[ind];
            }
          if (nn%2 == 1)
            r0 += dg_coefficients[mid] * xmid;
          array_face[0] = r0 + r1;
          array_face[1] = r0 - r1;
        }
      if (do_dg>1)
        {
          Number r0 = dg_coefficients[nn] * xp[0];
          Number r1 = dg_coefficients[nn+nn-1] * xm[0];
          for (unsigned int ind=1; ind<mid; ++ind)
            {
              r0 += dg_coefficients[nn+ind] * xp[ind];
              r1 += dg_coefficients[nn+nn-1-ind] * xm[ind];
            }
          if (nn%2 == 1)
            r1 += dg_coefficients[nn+mid] * xmid;
          array_face[2] = r0 + r1;
          array_face[3] = r0 - r1;
        }
    }
  else // contract_over_rows == false
    {
      for (unsigned int col=0; col<mid; ++col)
        {
          Number r0 = coefficients_eo[col*offset]        * xp[0];
          Number r1 = coefficients_eo[(nn-1-col)*offset] * xm[0];

          for (unsigned int ind=1; ind<mid; ++ind)
            {
              r0 += coefficients_eo[col*offset+ind]        * xp[ind];
              r1 += coefficients_eo[(nn-1-col)*offset+ind] * xm[ind];
            }
          if (nn % 2 == 1 && type > 0)
            r0 += coefficients_eo[col*offset+mid] * xmid;

          // dg contribution
          if (do_dg > 0)
            {
              r1 += dg_coefficients[col] * array_face[0];
              r0 += dg_coefficients[nn-1-col] * array_face[1];
            }
          if (do_dg > 1)
            {
              r0 += dg_coefficients[nn+col] * array_face[2];
              r1 += dg_coefficients[nn+nn-1-col] * array_face[3];
            }

          Number t = r0;
          r0 = (add_into ? array_for_add[col*stride] + t : t) + r1;
          if (type != 1)
            r1 = (add_into ? array_for_add[(nn-1-col)*stride] + t : t) - r1;
          else
            r1 = (add_into ? array_for_add[(nn-1-col)*stride] + r1 : r1) - t;
          if (nontemporal_store == false)
            {
              out[col*stride]        = r0;
              out[(nn-1-col)*stride] = r1;
            }
          else
            {
              r0.streaming_store(&out[col*stride][0]);
              r1.streaming_store(&out[(nn-1-col)*stride][0]);
            }
        }
      if (nn % 2 == 1)
        {
          Number r0 = (add_into ?
                       array_for_add[mid*stride] + coefficients_eo[mid*offset+mid] * xmid :
                       coefficients_eo[mid*offset+mid]*xmid);
          if (type == 1 && add_into)
            r0 = array_for_add[mid*stride];
          else if (type == 1)
            r0 = Number();

          if (type == 1)
            for (unsigned int ind=0; ind<mid; ++ind)
              r0 += coefficients_eo[mid*offset+ind] * xm[ind];
          else
            for (unsigned int ind=0; ind<mid; ++ind)
              r0 += coefficients_eo[mid*offset+ind] * xp[ind];

          // dg contribution
          if (do_dg > 0)
            r0 += dg_coefficients[mid] * array_face[0];
          if (do_dg > 1)
            r0 += dg_coefficients[nn+mid] * array_face[3];

          if (nontemporal_store == false)
            out[mid*stride] = r0;
          else
            r0.streaming_store(&out[mid*stride][0]);
        }
    }
}


#endif
