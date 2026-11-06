#!/bin/bash
# Pins the OpenMP threads of an unbound rank to its own block of cores:
# local rank r gets cores r*t .. r*t+t-1 (t = OMP_NUM_THREADS). Used when
# OpenMPI cannot bind the rank itself (it would span both sockets).
# Note: on Vanda cn nodes the CPU ids alternate between sockets (even ids on
# NUMA 0, odd on NUMA 1), so a block of consecutive ids spans both sockets.
r=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
t=${OMP_NUM_THREADS:-1}
export OMP_PLACES="{$((r * t))}:$t"
exec "$@"
