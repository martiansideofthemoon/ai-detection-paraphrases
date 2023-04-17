#!/bin/sh
#SBATCH --job-name=job_<exp_id>_<local_rank>
#SBATCH -o dipper_paraphrases/parallel/parallel_logs/logs_exp_<exp_id>/log_<local_rank>.txt
#SBATCH --partition=<gpu>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=<memory>
#SBATCH -d singleton
#SBATCH -t 48:00:00
<extra_args>

cd /work/kalpeshkrish_umass_edu/dipper_paraphrases

<command> --local_rank <local_rank> --num_shards <total>
