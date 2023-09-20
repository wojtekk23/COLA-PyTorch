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

# source /home/wk406185/PSP/COLA-PyTorch/venv/bin/activate
# cd /home/wk406185/PSP/COLA-PyTorch/
cat train.sh
# python main.py --audioset_train_folder=/home/user/data/train_wav --audioset_valid_folder=/home/user/data/valid_wav --output_size=1024 --save_every=1 run-25-07-2023/
python main.py --audioset_train_folder=/mnt/vdb/audioset-large/train_wav_16k --audioset_valid_folder=/mnt/vdb/audioset-large/valid_wav_16k --output_size=1024 --save_every=1 /mnt/vdb/run-30-08-2023/
# python main.py --audioset_train_folder=/mnt/vdb/random_audios_patch_16k --audioset_valid_folder=/mnt/vdb/random_audios_patch_16k_val2 --output_size=1024 --save_every=1 --finetune /mnt/vdb/run-finetune-30-08-2023/

