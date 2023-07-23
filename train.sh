#!/bin/bash
#
#SBATCH --job-name=train_cola
#SBATCH --partition=common
#SBATCH --qos=4gpu7d
#SBATCH --gres=gpu:1
##SBATCH --time=D-HH:MM:SS
#SBATCH --output=results/train_cola.txt
#SBATCH --mail-user=wojtek.klopotek+sbatch@gmail.com
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=END

source /home/wk406185/PSP/COLA-PyTorch/venv/bin/activate
cd /home/wk406185/PSP/COLA-PyTorch/
cat train.sh
python main.py --audioset_train_folder=/scidatalg/wk406185/audioset/train_wav --audioset_valid_folder=/scidatalg/wk406185/audioset_valid/valid_wav --save_every=50 run-22-07-2023/

