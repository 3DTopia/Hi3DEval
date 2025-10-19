export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

torchrun --nnodes=${NNODE} \
--nproc_per_node=${NUM_GPUS} \
--rdzv_backend=c10d \
$(dirname $0)//pretrain.py \
$(dirname $0)/config.py
