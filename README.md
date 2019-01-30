# Multigrid experiments with the deal.II library

This repository contains a collection of example programs demonstrating matrix-free multigrid solvers with the [deal.II finite element library](https://github.com/dealii/dealii). The code in this collection is based on an extension of the [step-37 tutorial program of deal.II](https://www.dealii.org/developer/doxygen/deal.II/step_37.html).

### Set of experiments

This repository contains the following experiments:

* poisson_cube: Solves the Poisson equation on a cube, running both a full multigrid cycle and a conjugate gradient solver preconditioned by a V-cycle.

* poisson_shell: Solves the Poisson equation with strongly varying coefficient on a shell, running both a full multigrid cycle and a conjugate gradient solver preconditioned by a V-cycle.

* poisson_l: Solves the Poisson equation on an L-shaped domain with a corner singularity using adaptive meshes.

* minimal_surface: Solves the nonlinear minimal surface equation, which is a particular variable-coefficient Poisson equation

#### GPU support

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
