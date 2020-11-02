#!/usr/bin/bash

# verify that argument (number of jobs) is passed
if [ -z "$1" ]
then
  echo "ERROR: number of jobs needs to be passed as an argument"
  exit 1
fi

# verify that argument (partition id) is passed
if [ -z "$2" ]
then
  echo "ERROR: number of partitions to use needs to be passed as an argument"
  exit 1
fi

# verify that only 2 arguments are passed
if [ $# -gt 2 ]
then
  echo "ERROR: $# arguments are given, only 2 are required"
  exit 1
fi

# verify that number of jobs is > 0
if [ $1 -lt 1 ]
then
  echo "ERROR: number of jobs is < 1"
  exit 1
fi

# verify that number of partitions is not > 3
if [ $2 -gt 3 ]
then
  echo "ERROR: partition id is > 3"
  exit 1
fi

# verify that number of partitions is >= 0
if [ $2 -lt 0 ]
then
  echo "ERROR: partition id is < 0"
  exit 1
fi

# define arrays
declare -a partitions=("tue.default.q"
                       "elec.default.q"
                       "elec.gpu.q"
                       "elec-em.gpu.q")
declare -a cpus=(8 8 4 10)
declare -a gpus=(0 0 1 1)

# execute jobs
for (( job_id=0; job_id<$1; job_id++ ))
do
  export job_id=$job_id
  export partition_id=$2
  export n_cpus=${cpus[$2]}
  export n_gpus=${gpus[$2]}
  sbatch  --job-name=sarnet_$job_id\_$2 \
          --nodes=1 \
          --ntasks=1 \
          --cpus-per-task=${cpus[$2]} \
          --gres=gpu:${gpus[$2]}\
          --time=10-00:00:00 \
          --partition=${partitions[$2]} \
          --output=o_$job_id\_$2.txt \
          --error=e_$job_id\_$2.txt \
          --mail-user=d.m.n.v.d.vorst@student.tue.nl \
          --mail-type=ALL \
          task.sh
  sleep 1
done