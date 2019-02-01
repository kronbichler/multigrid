# Multigrid experiments with the deal.II library

This repository contains a collection of example programs demonstrating matrix-free multigrid solvers with the [deal.II finite element library](https://github.com/dealii/dealii). The code in this collection is based on an extension of the [step-37 tutorial program of deal.II](https://www.dealii.org/developer/doxygen/deal.II/step_37.html).

### Set of experiments

This repository contains the following experiments:

* poisson_cube: Solves the Poisson equation on a cube, running both a full multigrid cycle and a conjugate gradient solver preconditioned by a V-cycle.

* poisson_shell: Solves the Poisson equation with strongly varying coefficient on a shell, running both a full multigrid cycle and a conjugate gradient solver preconditioned by a V-cycle. Note that Kronbichler and Wall (2018) report a relatively low number of iterations on a similar example compared to what is observed here, which is due to a particular analytic solution.

* poisson_l: Solves the Poisson equation on an L-shaped domain with a corner singularity using adaptive meshes.

* minimal_surface: Solves the nonlinear minimal surface equation, which is a particular variable-coefficient Poisson equation

### Run an experiment

To run an experiment, you need to first configure the test case against a deal.II installation. To obtain a compiled version of deal.II, see at the bottom of this page. Go into a directory, e.g. poisson_cube, and run the following:
```
cmake -D DEAL_II_DIR=/path/to/dealii/install -D CMAKE_BUILD_TYPE=Release .
```
The build type defines that we intend to run the program in release mode (with compiler optimizations and without debugging information). Whenever you do development of the program, it is recommended to use a Debug build (`-D CMAKE_BUILD_TYPE=Debug`) because deal.II has a large number of assertions that help to track down errors.

Running the program e.g. on a 12-core Intel Xeon E5-2687W v4 (Broadwell), the following output can be obtained for the cube example:
```
$ mpirun -n 12 -bind-to core ./program 4 150000000 2 2 2 square
Settings of parameters:
Polynomial degree:              4
Maximum size:                   150000000
Number of MG cycles in V-cycle: 2
Number of pre-smoother iters:   2
Number of post-smoother iters:  2
Use doubling mesh:              0

Testing FE_Q<3>(4)
Cycle 0
Number of degrees of freedom: 125 = (1 x 4 + 1)^3
Total setup time:      0.00313138s
Time compute rhs:      0.00776478
Time initial smoother: 4.5829e-05
Memory stats [MB]: 126.004 [p2] 126.341 128.035 [p8]
Time solve   (CPU/wall)    0.002141s/0.00853511s
Time solve   (CPU/wall)    9.8e-05s/0.000154858s
Time solve   (CPU/wall)    0.0001s/0.00015684s
Time solve   (CPU/wall)    0.000152s/0.000157359s
Time solve   (CPU/wall)    0.000152s/0.000155314s
All solver time 0.0091177 [p0] 0.00919578 0.00922101 [p2]
Coarse solver 12 times: 0.00920122 tot prec 0
level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC
Coarse solver 3 times: 1.6763e-05 tot prec 0.00040683
level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC
matvec time dp 3.18e-07 [p9] 1.3557e-06 8.1292e-06 [p11] DoFs/s: 1.5377e+07
matvec time dp 3.1774e-07 [p9] 1.3338e-06 7.9705e-06 [p11] DoFs/s: 1.5683e+07
matvec time dp 3.1548e-07 [p9] 1.3489e-06 7.9662e-06 [p11] DoFs/s: 1.5691e+07
matvec time dp 3.1542e-07 [p9] 1.3236e-06 7.926e-06 [p11] DoFs/s: 1.5771e+07
matvec time dp 3.148e-07 [p2] 1.2615e-06 7.3561e-06 [p11] DoFs/s: 1.6993e+07
Best timings for ndof = 125   mv 7.3561e-06    mv smooth 7.5114e-06   mg 0.00014986
L2 error with ndof = 125  1.1797  with CG 1.1797

...


Cycle 23
Number of degrees of freedom: 135005697 = (128 x 4 + 1)^3
Total setup time:      11.333s
Time compute rhs:      8.977
Time initial smoother: 5.9028
Memory stats [MB]: 2347.3 [p3] 2375.3 2428.2 [p1]
Time solve   (CPU/wall)    8.0494s/8.3143s
Time solve   (CPU/wall)    3.6149s/3.6178s
Time solve   (CPU/wall)    3.6129s/3.6138s
Time solve   (CPU/wall)    3.6099s/3.6128s
Time solve   (CPU/wall)    3.6154s/3.6163s
error start         level 1: 1.3006
residual norm start level 1: 27.4
residual norm end   level 1: 0.32679
error end           level 1: 0.17372
error start         level 2: 0.17468
residual norm start level 2: 9.5456
residual norm end   level 2: 0.24821
error end           level 2: 0.011664
error start         level 3: 0.015083
residual norm start level 3: 0.73235
residual norm end   level 3: 0.01274
error end           level 3: 0.00040369
error start         level 4: 0.00051142
residual norm start level 4: 0.03154
residual norm end   level 4: 0.00040747
error end           level 4: 1.268e-05
error start         level 5: 1.6539e-05
residual norm start level 5: 0.0013008
residual norm end   level 5: 2.0988e-05
error end           level 5: 4.2626e-07
error start         level 6: 5.4339e-07
residual norm start level 6: 5.5035e-05
residual norm end   level 6: 1.0401e-06
error end           level 6: 1.3769e-08
error start         level 7: 1.7362e-08
residual norm start level 7: 2.4024e-06
residual norm end   level 7: 4.7265e-08
error end           level 7: 4.342e-10
All solver time 22.774 [p5] 22.775 22.775 [p1]
Coarse solver 96 times: 0.02376 tot prec 0
level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC
L1     0.003677    0.0007593 0.0002591 0.0008119 0.0009744   7.357e-05
L2     0.004806    0.00097   0.0002615 0.001109  0.0009335   2.076e-05
L3     0.01286     0.002621  0.0004627 0.001674  0.001217    3.812e-05
L4     0.06815     0.01343   0.001536  0.004285  0.004828    0.0001227
L5     0.5057      0.07605   0.02739   0.02451   0.03663     0.0005227
L6     3.209       0.432     0.2977    0.1762    0.3602      0.002975
L7     14.74       2.066     1.749     0.6926    1.824       0.01259
Coarse solver 8 times: 0.00024006 tot prec 11.166
level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC
L1     0.003804    6.966e-05 1.968e-05 0.0003543 9.869e-05   0
L2     0.003023    0.000139  1.973e-05 0.0002264 0.000119    0
L3     0.003226    0.0003972 2.809e-05 0.0004738 0.0001548   0
L4     0.0183      0.002385  0.000139  0.002703  0.0007166   0
L5     0.1235      0.02776   0.002822  0.008956  0.00642     0
L6     0.8902      0.1219    0.04046   0.0897    0.08435     0
L7     6.732       0.7539    1.08      0.5136    0.6526      0
matvec time dp 0.15706 [p10] 0.15726 0.15748 [p0] DoFs/s: 8.5731e+08
matvec time dp 0.15469 [p8] 0.15473 0.15475 [p1] DoFs/s: 8.724e+08
matvec time dp 0.15448 [p8] 0.15452 0.15455 [p4] DoFs/s: 8.7355e+08
matvec time dp 0.15477 [p11] 0.15482 0.15486 [p1] DoFs/s: 8.7177e+08
matvec time dp 0.15488 [p8] 0.1549 0.15494 [p1] DoFs/s: 8.7135e+08
Best timings for ndof = 135005697   mv 0.15455    mv smooth 0.092618   mg 3.6128
L2 error with ndof = 135005697  4.342e-10  with CG 4.2068e-10

Cycle 24
Number of degrees of freedom: 263374721 = (160 x 4 + 1)^3
Total setup time:      19.549s
Max size reached, terminating.

 cells    dofs    mv_outer  mv_inner  reduction  fmg_L2error   fmg_time    cg_L2error    cg_time  cg_its cg_reduction
1       125       7.356e-06 7.511e-06 1.000e+00 1.180e+00 -    1.499e-04 1.180e+00 -    9.667e-04 3      1.262e-04
8       729       1.264e-05 6.969e-06 1.092e-01 1.737e-01 2.76 7.675e-04 1.725e-01 2.77 2.569e-03 8      5.677e-02
27      2197      9.923e-06 8.709e-06 1.000e+00 4.102e-02 3.56 6.810e-04 4.102e-02 3.54 1.234e-03 3      3.157e-04
64      4913      1.264e-05 9.717e-06 1.613e-01 1.166e-02 4.37 8.097e-04 1.027e-02 4.81 2.154e-03 8      6.789e-02
125     9261      1.882e-05 1.528e-05 1.000e+00 3.902e-03 4.91 1.971e-03 3.902e-03 4.34 3.241e-03 3      4.421e-04
216     15625     3.129e-05 2.267e-05 1.818e-01 2.164e-03 3.23 1.813e-03 1.145e-03 6.73 4.744e-03 8      6.134e-02
343     24389     3.721e-05 2.675e-05 1.000e+00 7.057e-04 7.27 4.772e-03 7.057e-04 3.14 7.448e-03 3      5.026e-04
512     35937     5.361e-05 3.688e-05 1.319e-01 4.037e-04 4.18 2.032e-03 3.822e-04 4.59 5.333e-03 8      6.689e-02
1000    68921     1.007e-04 6.069e-05 1.202e-01 1.246e-04 5.27 4.994e-03 1.316e-04 4.78 1.297e-02 8      6.575e-02
1728    117649    1.611e-04 1.103e-04 1.250e-01 5.413e-05 4.57 4.690e-03 5.423e-05 4.86 1.251e-02 8      6.828e-02
2744    185193    2.365e-04 1.588e-04 1.142e-01 2.418e-05 5.23 1.249e-02 2.546e-05 4.90 3.190e-02 8      6.722e-02
4096    274625    3.513e-04 2.322e-04 1.137e-01 1.268e-05 4.83 7.570e-03 1.319e-05 4.93 2.336e-02 8      6.948e-02
8000    531441    5.852e-04 3.784e-04 1.163e-01 4.252e-06 4.90 1.565e-02 4.370e-06 4.95 4.683e-02 8      6.989e-02
13824   912673    9.802e-04 6.299e-04 1.196e-01 1.749e-06 4.87 2.106e-02 1.767e-06 4.97 7.399e-02 8      7.029e-02
21952   1442897   1.672e-03 9.588e-04 1.238e-01 8.213e-07 4.90 4.311e-02 8.207e-07 4.98 1.383e-01 8      6.970e-02
32768   2146689   2.598e-03 1.489e-03 1.270e-01 4.263e-07 4.91 5.220e-02 4.220e-07 4.98 2.027e-01 8      6.985e-02
64000   4173281   5.452e-03 3.194e-03 1.318e-01 1.418e-07 4.93 1.151e-01 1.387e-07 4.99 4.449e-01 8      6.957e-02
110592  7189057   8.965e-03 5.382e-03 1.346e-01 5.748e-08 4.95 1.906e-01 5.582e-08 4.99 7.639e-01 8      6.917e-02
175616  11390625  1.413e-02 8.487e-03 1.363e-01 2.675e-08 4.96 3.182e-01 2.585e-08 4.99 1.232e+00 8      6.894e-02
262144  16974593  2.054e-02 1.245e-02 1.375e-01 1.377e-08 4.97 4.564e-01 1.327e-08 5.00 1.806e+00 8      6.863e-02
512000  33076161  4.044e-02 2.408e-02 1.389e-01 4.533e-09 4.98 9.125e-01 4.351e-09 5.00 3.555e+00 8      6.839e-02
884736  57066625  6.685e-02 4.031e-02 1.396e-01 1.826e-09 4.99 1.538e+00 1.750e-09 5.00 6.023e+00 8      6.826e-02
1404928 90518849  1.054e-01 6.306e-02 1.400e-01 8.459e-10 4.99 2.452e+00 8.102e-10 4.99 9.551e+00 8      6.807e-02
2097152 135005697 1.545e-01 9.262e-02 1.403e-01 4.342e-10 4.99 3.613e+00 4.207e-10 4.91 1.649e+01 8      6.799e-02

```

The final table displays a lot of information, such as the time for matrix-vector products within the conjugate gradient method and correction (mv_outer, done in double precision with the default settings) or for the matrix-vector product within the smoother (mv_inner, done in single precision with the default settings). Furthermore, the table contains numbers about the convergence rate of the multigrid iteration and the L2 discretization error after the multigrid solution, both for the full multigrid cycle (FMG) and the conjugate gradient method.

### GPU support

The above experiments have also been run a GPU, using the preliminary code available here: https://github.com/kalj/dealii-cuda/. The deal.II library has preliminary support for GPU computations, but the feature is still work in progress. The above examples are mostly compatible with the CUDA infrastructure and will be added to the list soon. Please open an issue for a particular setup.

#### Related manuscripts

* [M. Kronbichler and W. A. Wall (2018)](https://epubs.siam.org/doi/10.1137/16M110455X). A performance comparison of continuous and discontinuous Galerkin methods with fast multigrid solvers. *SIAM J. Sci. Comput.*, 40(5):A3423-A3448. Relates to poisson_cube and poisson_shell.

* [K. Ljungkvist and M. Kronbichler (2017)](http://www.it.uu.se/research/publications/reports/2017-006/). Multigrid for Matrix-Free Finite Element Computations on Graphics Processors. Technical Report 2017-006, Department of Information Technology, Uppsala University. Relates to poisson_shell, poisson_l, minimal_surface.

### Installation of the deal.II package

This suite consists of small programs that run against a compiled deal.II finite element library that in turn needs a C++ compiler adhering to the C++11 standard, an MPI implementation, and cmake. The following software packages are needed:

* deal.II, using at least version 9.0.0, see www.dealii.org. deal.II must be configured to also include the following external packages (no direct access to these packages is necessary, except for the interface through deal.II):

* MPI

* p4est for providing parallel adaptive mesh management on forests of quad-trees (2D) or oct-trees (3D). For obtaining p4est, see http://www.p4est.org. p4est of at least version 2.0 is needed for running this project. Installation of p4est can be done via a script provided by deal.II (the first argument refers to the path of the deal.II source directory as obtained by cloning from github.com/dealii/dealii, the last argument specifies the desired installation directory for p4est):
```
/path/to/dealii/doc/external-libs/p4est-setup.sh p4est-1.1.tar.gz /path/to/p4est/install
```

Given these dependencies, the configuration of deal.II in a subfolder `/path/to/dealii/build` can be done through the following commands:
```
cmake \
    -D CMAKE_CXX_FLAGS="-march=native" \
    -D CMAKE_INSTALL_PREFIX="/path/to/dealii/install/" \
    -D DEAL_II_WITH_MPI="ON" \
    -D DEAL_II_WITH_LAPACK="ON" \
    -D DEAL_II_WITH_P4EST="ON" \
    -D P4EST_DIR="/path/to/p4est/install/" \
    ../deal.II
```

Since the matrix-free algorithms in deal.II make intensive use of advanced processor instruction sets (e.g. vectorization through AVX/AVX-512 or similar), it is recommended to enable processor-specific optimizations (second line, `-march=native`). The path on the third line specifies the desired installation directory of deal.II, and the last line points to the location of the source code of deal.II relative to the folder where the cmake script is run. After configuration, run

```
make -j8
make install
```

to compile deal.II and install it in the given directory. After installation, the deal.II sources and the build folder are no longer necessary (unless you find bugs in deal.II and need to modify that code). It is also possible to run against a deal.II build folder, say `/path/to/dealii/build`, that then combines with the include folders in the source directory.

### Literature

The matrix-free infrastructure in deal.II is described in the following papers:

* [M. Kronbichler and K. Kormann (2012)](https://doi.org/10.1016/j.compfluid.2012.04.012). A generic interface for parallel cell-based finite element operator application. *Comput. Fluids*, 63:135-147.

* [M. Kronbichler and K. Kormann (2017)](https://arxiv.org/abs/1711.03590). Fast matrix-free evaluation of discontinuous Galerkin finite element operators. *Preprint* arXiv:1711.03590.
