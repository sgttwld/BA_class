# BA_class

Implementation of a collection of Blahut-Arimoto type algorithms for information-theoretic bounded rationality for easy access. The class contains one-step rate distortion, two-step parallel and serial cases, as developed in  [Genewein et al. Bounded Rationality, Abstraction, and Hierarchical Decision-Making: An Information-Theoretic Optimality Principle](https://doi.org/10.3389/frobt.2015.00027), an arbitrary depth version of the parallel case, and additionally the option for a bound on the prior adaption.

## Simple examples

In order to do a one-step rate distortion iteration, we write
```python
RD = BA.Solver([N,K],beta=[1.0])
RD.iterate(U)
```
where `N` and `K` are the dimensions of the world and action space, respectively, and `U` is the utility function in form of an `N`times`K` matrix (`numpy` array of shape `(N,K)`).

For a two-step iteration with intermediate dimension `M` we write
```python
ser = BA.Solver([N,M,K],beta=[beta1,beta2],BAtype='ser')    # serial case
ser.iterate(U)

par = BA.Solver([N,M,K],beta=[beta1,beta2],BAtype='par')    # parallel case
par.iterate(U)
```

See also [here](https://github.com/sgttwld/blahut-arimoto) for a different implementation of some of the cases included here based on [pr_func](https://github.com/sgttwld/pr_func).


## Overview

### Usage
```python
import BA_class as BA                         # import the module
dims = [N,M1,...,Mn,K]                        # dimensions (world,...intermediate...,action)
par = BA.Solver(dims,beta=beta,BAtype='par')  # beta= [beta1,...], len(beta) = number of steps
par = BA.Solver(dims,beta=beta,alpha=alpha,BAtype='par')  # alpha = [alpha1,...] for bounded priors
par.iterate(U)                                # iteration with utility U
```

### Interesting quantities
```python
par.EU            # Expected utility
par.FE            # Free Energy
par.DKL           # Kullback-Leibler divergences between priors and posteriors
par.DKLpr         # Kullback-Leibler divergences between fixed uniform priors and the priors
par.joint         # the joint distribution as a numpy array
par.pagw          # resulting total policy p(a|w) (intermediate variables marginalized out)
par.post          # list of all posteriors
par.prior         # list of all priors (exception: prior[0] is the world state distribution)
```

### Special options
```python
par.iterate(U,max_iterations=10000)           # maximum number of iterations
par.iterate(U,precision=1e-10)                # precision for the stopping condition
par.iterate(U,pw=np.array([]))                # specification of the world state distribution
par.iterate(U,p0=np.array([]))                # fixed prior for the prior restriction
```

### Other methods
```python
BA.show(arr)                                  # plots the array arr (1d: barplot, 2d: pcolor)
```
