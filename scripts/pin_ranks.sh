#!/bin/bash
# Pins MPI rank r to cores [r*t, (r+1)*t), t = OMP_NUM_THREADS. Use with
# `mpirun --bind-to none` where OpenMPI's own --bind-to core is unreliable
# (it aborted beyond 13 ranks in our container).
t=${OMP_NUM_THREADS:-1}
r=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
first=$((r * t))
exec taskset -c "$first-$((first + t - 1))" "$@"
