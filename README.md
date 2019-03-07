# UCP-Solver

The repository provides the source codes of UCP solver provided in the paper

S.-H. Tseng, "A Generic Solver for Unconstrained Control Problems." 

## Requirements
Two versions of the solver are included: the single threaded version and the parallelized version. For compilation, the single threaded version requires g++ compiler and c++ standard libraries; The parallelized version requires NVIDIA CUDA compiler and libraries. GNU Make is necessary for compilation automation.

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

## Customization
To extend the solver to the customized problems, add a new file under the "main" folder. One can make a copy of the exising examples in "main" and change the content.

The customization involves inheriting the base class UCPSolver, which is defined in UCP_solver.h under "sources." The user can read witsenhausen.h to learn how to extend UCPSolver to a customized problem.