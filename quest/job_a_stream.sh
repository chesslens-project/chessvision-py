#!/bin/bash
#SBATCH --account=p32731
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=stream_a
#SBATCH --output=/projects/p32731/chessvision/logs/stream_a_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rakkshet.singhaal@kellogg.northwestern.edu
echo "Job A started: $(date)"
module load python/3.12.10
pip install chess gensim zstandard tqdm requests --quiet --user
PROJECT=/projects/p32731/chessvision
mkdir -p $PROJECT/tokens $PROJECT/logs
python3 $PROJECT/lichess_stream.py --start 2013-01 --end 2019-12 --output $PROJECT/tokens
echo "Job A finished: $(date)"
ls $PROJECT/tokens/*.txt.gz | wc -l
