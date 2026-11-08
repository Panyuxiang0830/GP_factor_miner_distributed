# Sourced by the Vanda job scripts (run from the repository root).
#   GP_DATA  exported panels and expressions (default /scratch/$USER/gp_data)
#   OUT      CSV results (default /scratch/$USER/gp_runs/<date>)
#   GP_EXPRS expression batch (default: the selected factor library)
module load OpenMPI/5.0.8-GCC-14.3.0 CMake/4.0.3-GCCcore-14.3.0 >/dev/null 2>&1

REPO=$PWD
BIN=$REPO/build_vanda
DATA=${GP_DATA:-/scratch/$USER/gp_data}
PANELS=$DATA/GP_distributed/evolution_full
EXPRS=${GP_EXPRS:-$DATA/GP_output_eod/rescreen_rankic_007_corr_075_unlimited/selected_gp_factors.json}
OUT=${OUT:-/scratch/$USER/gp_runs/$(date +%Y%m%d)}
mkdir -p "$OUT"

# This OpenMPI starts its daemons over ssh (no PBS TM); the launch agent loads
# the modules for them on each node.
export PRTE_MCA_prte_launch_agent=$REPO/scripts/vanda/prted.sh
export OMP_PLACES=cores OMP_PROC_BIND=close

HOSTS=$(sort -u "$PBS_NODEFILE" | wc -l)
# Nodes of the CPU pool differ (CPU-only and A40 nodes); record which ones ran.
for h in $(sort -u "$PBS_NODEFILE"); do
    echo "== host $h: $(ssh "$h" lscpu 2>/dev/null | grep 'Model name' | sed 's/.*: *//')"
done

# mpirun placement of R ranks with t threads each on h hosts. OpenMPI refuses
# to bind one rank to cores of both sockets (36 each), and PE=t always binds,
# so when some rank would straddle the sockets (t > 36, or an odd number of
# ranks per node that do not divide a socket, e.g. 9 x 8) the ranks are placed
# without PE and left unbound; omp_slice.sh then pins each rank's threads to
# its own block of cores.
straddles() {
    local R=$1 t=$2 h=$3
    [ "$t" -gt 36 ] || { [ $((36 % t)) -ne 0 ] && [ $((R / h * t)) -gt 36 ]; }
}
placement() {
    if straddles "$@"; then
        echo "--map-by ppr:$(($1 / $3)):node --bind-to none"
    else
        echo "--map-by ppr:$(($1 / $3)):node:PE=$2 --bind-to core"
    fi
}
# Command prefix that pins the threads of unbound ranks.
pin() { straddles "$@" && echo "$REPO/scripts/vanda/omp_slice.sh"; }

# bench <ranks> <threads per rank> <hosts used> [gpe_bench options...]
# Places ranks/hosts ranks on each host, each bound to <threads> cores.
bench() {
    local R=$1 t=$2 h=$3
    shift 3
    echo "-- R=$R t=$t hosts=$h $*"
    mpirun -np "$R" $(placement "$R" "$t" "$h") \
        -x OMP_NUM_THREADS="$t" -x OMP_PLACES -x OMP_PROC_BIND \
        $(pin "$R" "$t" "$h") "$BIN/gpe_bench" "$PANELS" "$EXPRS" "$@" || echo "FAILED: R=$R t=$t hosts=$h $*"
}

# consistency <ranks> <threads> <hosts> <H0> <exprs> <op|node> [chunk]
consistency() {
    local R=$1 t=$2 h=$3
    echo "-- consistency R=$R t=$t hosts=$h H0=$4 exprs=$5 $6"
    mpirun -np "$R" $(placement "$R" "$t" "$h") \
        -x OMP_NUM_THREADS="$t" -x OMP_PLACES -x OMP_PROC_BIND \
        $(pin "$R" "$t" "$h") "$BIN/test_mpi_consistency" "$PANELS" "$EXPRS" "$4" "$5" "$6" ${7:+"$7"} || echo "FAILED: consistency R=$R"
}
