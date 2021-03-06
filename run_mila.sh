#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=16:00:00
#SBATCH --exclude=kepler2


echo "running on $SLURMD_NODENAME"
export PYTHONUNBUFFERED=1

source ~/pytorchenv/bin/activate
MYPYTHON_PATH=`which python`
echo "Activate my env, curr python ${MYPYTHON_PATH}"
python -c "import torch; torch.zeros(1).cuda()"
if [ $? -eq 0 ]
then
  echo "Test cuda OK!"
else
  echo "Test cuda FAILED!"
  exit
fi

# Unzip data
echo "Get data..."
tar -xzf ./cls2_data/data.tar.gz -C $SLURM_TMPDIR

echo "$@"
exec $@
