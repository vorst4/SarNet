#!/usr/bin/bash
module load cuda10.2/toolkit/10.2.89
module load cudnn7.6-cuda10.1/7.6.5.32
source /home/tue/s111167/python-env/main/bin/activate
python main.py --job_id $job_id \
               --partition_id $partition_id \
               --n_cpus $n_cpus \
               --n_gpus $n_gpus