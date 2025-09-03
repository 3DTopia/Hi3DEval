MASTER_PORT=$(shuf -i 10000-19999 -n 1)
MASTER_ADDR="127.0.0.1"
echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "which python: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "PYTHONPATH: ${PYTHONPATH}"

NNODE=1
NUM_GPUS=1

torchrun --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    $(dirname $0)/inference.py \
    $(dirname $0)/config.py
