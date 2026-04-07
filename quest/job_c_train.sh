#!/bin/bash
#SBATCH --account=p32731
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=96:00:00
#SBATCH --mem=128GB
#SBATCH --job-name=chess2vec_train
#SBATCH --output=/projects/p32731/chessvision/logs/train_%j.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rakkshet.singhaal@kellogg.northwestern.edu
echo "Job C started: $(date)"
module load python/3.12.10
pip install chess gensim zstandard tqdm requests --quiet --user
PROJECT=/projects/p32731/chessvision
echo "Token files available:"
ls $PROJECT/tokens/*.txt.gz | wc -l
du -sh $PROJECT/tokens/
python3 $PROJECT/train_chess2vec.py --tokens $PROJECT/tokens --output $PROJECT/models --vector-size 128 --window 5 --workers $SLURM_CPUS_PER_TASK --epochs 10
echo "Job C finished: $(date)"
ls -lh $PROJECT/models/
