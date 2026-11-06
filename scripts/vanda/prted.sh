#!/bin/bash
# Launch agent for mpirun on Vanda. OpenMPI 5 starts its PRRTE daemons over
# ssh, and a non-interactive ssh shell has no modules loaded, so load them here.
#   export PRTE_MCA_prte_launch_agent=$REPO/scripts/vanda/prted.sh
source /etc/profile.d/modules.sh 2>/dev/null || source /etc/profile >/dev/null 2>&1
module load OpenMPI/5.0.8-GCC-14.3.0 >/dev/null 2>&1
exec prted "$@"
