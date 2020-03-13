# UCP-Solver

The repository provides the source codes of UCP solver provided in the paper

S.-H. Tseng, "A Generic Solver for Unconstrained Control Problems with Integral Functional Objectives," in Proc. IEEE ACC 2020. 

## Requirements
Two versions of the solver are included: the single threaded version and the parallelized version. For compilation, the single threaded version requires g++ compiler and c++ standard libraries; The parallelized version requires NVIDIA CUDA compiler and libraries. GNU Make is necessary for compilation automation. The programs are tested under Linux.

## Compilation and Execution
To compile each version, get into each folder and run
```sh
$ make
```

To run the programs, use
```sh
$ ./executables/<program name>
```

For instance, to run the example given by the source code <main_witsenhausen.cpp> under the "main" folder, type
```sh
$ ./executables/main_witsenhausen
```
and the results can be collected under the "results" folder.

## Results
Once the solver converges under the given criterion, it will generate the result files correspondingly under results/. For example, the file "u0.dat" corresponds to the controller u_0 (y_0). The first column of the file is y_0 and the second column is u_0(y_0).

## Customization
To extend the solver to the customized problems, add a new file under the "main" folder. One can make a copy of the exising examples in "main" and change the content.

The customization involves inheriting the base class UCPSolver, which is defined in UCP_solver.h under "sources." The user can read witsenhausen.h to learn how to extend UCPSolver to a customized problem.