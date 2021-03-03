#!/usr/bin/env bash
source /home/humt_ma/.conda_remote
conda activate detectron2

echo "=====Job Infos ===="
echo "Node List: " "$SLURM_NODELIST"
echo "Job ID: " "$SLURM_JOB_ID"
echo "Job Name:" "$SLURM_JOB_NAME"
echo "Partition: " "$SLURM_JOB_PARTITION"
echo "Submit directory:" "$SLURM_SUBMIT_DIR"
echo "Submit host:" "$SLURM_SUBMIT_HOST"
echo "In the directory: $(pwd)"
echo "As the user: $(whoami)"
echo "Python version: $(python -c 'import sys; print(sys.version)')"
echo "pip version: $(pip --version)"

nvidia-smi

start_time=$(date +%s)
echo "Job Started at $(date)"

python /net/rmc-lx0114/home_local/git/detectron2/demo/justin_hyper.py --data case_no_alpha --path_prefix /net/rmc-gpu03/home_local/humt_ma --out_dir data/justin_training --base_config mask_rcnn --evals 100

echo "Job ended at $(date)"
end_time=$(date +%s)
total_time=$((end_time - start_time))
echo "Took " ${total_time} " s"