DYNETINF is a very basic python implementation of the topology
tracking algorithm, using cascade traces. Since it was primarilydeveloped for prototyping the algorithm, it may contain unknown bugs, and it includes some hard-coded parameters. A future stable and bug-free version is under development.

Dependent python libraries:
numpy, scipy, networkx

Data inputs:

dataset.p - contains a timeseries (T=1000) of synthetic cascade data as a serialized file

gtruth.p - contains a timeseries of ground-truth adjacency matrices as a serialized file

Python scripts:
test.py - calls the tracking algorithm

netinf/netinf.p - contains the tracking algorithm as a function


To run:

python test.py

Note:

The output is basic error output, and a matplotlib graph
showing the relative absolute error.

