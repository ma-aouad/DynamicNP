This directory provides an implementation of the methods for rank-based dynamic assortment optimization models.

## Dependencies

Julia 0.5

Distributions 0.13.0

NPZ 0.2.0

## Types

`Instance` is the user-instantiated type, describing the parameters of each generated instance of the dynamic assortment problem: number of products, customer types, distribution of the number of arrivals, and choice model.

## Functional description

`instancegen.jl`: methods relative to the instance generation, logs, and revenue evaluation.

`heuristics.jl`: implementation of heuristics and approximations including discrete-greedy, gradient-descent, and local search.

`approx.jl`: implementation of heuristics and approximations for the general rank-based model and nested choice model.

Usage examples are provided in `main.jl`. A benchmark of instances is provided in the folder `instances` and the corresponding method outputs are provided in the folder `outputs`.