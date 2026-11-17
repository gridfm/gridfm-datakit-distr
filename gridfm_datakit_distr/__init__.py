"""gridfm-datakit-distr: datasets for three-phase unbalanced distribution networks.

Power-flow labels are produced with OpenDSS (DSS-Extensions); OPF labels with Ipopt through
CasADi on the extracted admittances, verified in OpenDSS; cross-checks with
PowerModelsDistribution.jl.
"""

__version__ = "0.2.0.dev6"
