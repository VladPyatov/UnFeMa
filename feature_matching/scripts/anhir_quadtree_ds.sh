SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../"

# # conda activate anhir_loftr
# export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=512
data_cfg_path="configs/data/anhir_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/anhir/loftr_ds_quadtree.py"

n_nodes=1
n_gpus_per_node=2
torch_num_workers=4
batch_size=10
pin_memory=true
exp_name="outdoor-quadtree-anhir-ds-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"
ckpt_path="checkpoints/outdoor.ckpt"

python -u ./train_anhir.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="cuda" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=-1 \
    --benchmark=True \
    --max_epochs=64
   