#!/bin/bash
#SBATCH --mail-user=s_ssaina@live.concordia.ca
#SBATCH --mail-type=ALL

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

scp $HOME/projects/def-ravanelm/datasets/wsj0-2mix-8k-min.tar.gz $SLURM_TMPDIR/
cd $SLURM_TMPDIR
mkdir WSJ0Mix && tar -zxf wsj0-2mix-8k-min.tar.gz -C WSJ0Mix

module load StdEnv/2023  gcc/12.3 intel/2023.2.1 gcccore/.12.3 ucc/1.2.0 ucx/1.14.1 openmpi/4.1.5 arrow/17.0.0 cuda/11.8
source $HOME/projects/def-ravanelm/salmanhu/speechbrain/MambaEnv/bin/activate

cd $HOME/projects/def-ravanelm/salmanhu/speechbrain/recipes/WSJ0Mix/separation
python train_student.py distillation_hparams/resepformer/base_codecformer.yaml  --data_folder=$SLURM_TMPDIR/WSJ0Mix/wsj0-2mix-8k-min/wsj0-mix/2speakers
