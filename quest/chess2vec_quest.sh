#!/bin/bash
#SBATCH --account=p32731
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=chess2vec
#SBATCH --output=/projects/p32731/chessvision/logs/chess2vec_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vbw3216@u.northwestern.edu

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

module load python/3.12.10

PROJECT=/projects/p32731/chessvision
TOKENS=$PROJECT/tokens
MODEL=$PROJECT/models
LOGS=$PROJECT/logs

mkdir -p $TOKENS $MODEL $LOGS

pip install chess gensim zstandard tqdm requests --quiet --user

echo ""
echo "=== STEP 1: Streaming Lichess data ==="
python3 $PROJECT/lichess_stream.py \
    --months 3 \
    --output $TOKENS

echo "Token files:"
ls -lh $TOKENS/*.txt.gz

echo ""
echo "=== STEP 2: Training chess2vec ==="
python3 $PROJECT/train_chess2vec.py \
    --tokens  $TOKENS \
    --output  $MODEL \
    --vector-size 128 \
    --window  5 \
    --workers $SLURM_CPUS_PER_TASK \
    --epochs  10

echo ""
echo "Job finished: $(date)"
echo "Model files:"
ls -lh $MODEL/
