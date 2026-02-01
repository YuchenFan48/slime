#!/usr/bin/env bash
#

set -ex

echo "============================environment========================="
env | while IFS='' read -r line; do
  echo "$line"
done
echo "================================================================"

MODEL_PREFIX=$1
if [ -z "$MODEL_PREFIX" ]; then
  VERL_TRAIN_SCRIPT="/apdcephfs/mnt/cephfs/users/yuchenfan/slime/scripts/run-qwen3-next-4m.sh"
else
  VERL_TRAIN_SCRIPT="/apdcephfs/mnt/cephfs/users/yuchenfan/slime/scripts/run-qwen3-next-4m.sh"
fi


# for local test
if [[ -z "${__POD_IP__}" ]]; then
  readonly MASTER_ADDR="127.0.0.1"
else
  readonly MASTER_ADDR="$__POD_IP__" # gemini-2
fi
readonly MASTER_PORT="7689"
readonly RANK=0

cp /etc/mpi/hostfile .
# 使用sed命令修改文件内容
# sed -i 's/=8/=1/g' hostfile
HOSTFILE=`pwd`/hostfile
if [ -f $HOSTFILE ]; then
  readonly NNODES=`wc -l $HOSTFILE | awk '{print $1}'` # gemini-2
else
  readonly NNODES=1
fi

GEMINI_MPI_ARGS='--hostfile '$HOSTFILE' --bind-to none --map-by slot --mca routed direct --mca btl_tcp_if_include bond1 --mca oob_tcp_if_include bond1  -x PATH -x LIBRARY_PATH -x LD_LIBRARY_PATH'

echo $NNODES $MASTER_ADDR

run_cmd="mpirun \
  -v --allow-run-as-root -np $NNODES -N 1 \
  $GEMINI_MPI_ARGS"

exec_cmd="bash $VERL_TRAIN_SCRIPT $MASTER_ADDR $MASTER_PORT $NNODES "
${run_cmd} ${exec_cmd} &
