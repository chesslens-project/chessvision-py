#!/bin/bash
#SBATCH --account=YOUR_NETID
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=chess2vec
#SBATCH --output=/projects/YOUR_NETID/chessvision/logs/chess2vec_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@northwestern.edu

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

# Paths — edit these to match your Quest allocation
PROJECT=/projects/YOUR_NETID/chessvision
TOKENS=$PROJECT/tokens
MODEL=$PROJECT/models
LOGS=$PROJECT/logs

mkdir -p $TOKENS $MODEL $LOGS

# Load Python
module load python/3.12.6

# Install dependencies if needed
pip install chess gensim zstandard tqdm requests --quiet

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Download and stream 3 months of Lichess data
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== STEP 1: Streaming Lichess data ==="
python3 $PROJECT/lichess_stream.py \
    --months 3 \
    --output $TOKENS

echo "Token files:"
ls -lh $TOKENS/*.txt.gz

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Train chess2vec
# ─────────────────────────────────────────────────────────────────────────────
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
