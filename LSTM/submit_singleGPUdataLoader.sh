#!/bin/bash 
#SBATCH -C gpu 
#SBATCH --qos=regular
#SBATCH -A gc5_g
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 1
#SBATCH --time=12:00:00
#SBATCH --image=romerojosh/containers:sc21_tutorial # check with "shifterimg images"
#SBATCH -J pm-crop64
#SBATCH -o %x-%j.out

DATADIR=/pscratch/sd/h/hsko/jupyter/jupyter/ML/FCN/sound/audio_spectrogram/dataset/DSD100
LOGDIR=${SCRATCH}/jupyter/ML/LSTM/log
mkdir -p ${LOGDIR}
args="${@}"

hostname

module load pytorch/1.13.1 # <-handled by the shifter If ON: no shifter
pip install torchinfo
# pip install wandb

# I have no authority to install benchy in the node!
# Use shifter above... #SBATCH --image=romerojosh/containers:sc21_tutorial # check with "shifterimg images" 
# git clone https://github.com/romerojosh/benchy.git && \
# cd benchy && \
# python setup.py install && \
# cd ../ && rm -rf benchy

#~/dummy

export NCCL_NET_GDR_LEVEL=PHB

# Profiling
if [ "${ENABLE_PROFILING:-0}" -eq 1 ]; then
    echo "Enabling profiling..."
    NSYS_ARGS="--trace=cuda,cublas,nvtx --kill none -c cudaProfilerApi -f true"
    NSYS_OUTPUT=${PROFILE_OUTPUT:-"profile"}
    export PROFILE_CMD="nsys profile $NSYS_ARGS -o $NSYS_OUTPUT"
fi

BENCHY_CONFIG=benchy-conf.yaml
BENCHY_OUTPUT=${BENCHY_OUTPUT:-"benchy_output"} # It checks if the variable BENCHY_OUTPUT is set or not. 
# If it is set, it uses its value; otherwise, it uses the default value "benchy_output".
sed "s/.*output_filename.*/        output_filename: ${BENCHY_OUTPUT}.json/" ${BENCHY_CONFIG} > benchy-run-${SLURM_JOBID}.yaml
export BENCHY_CONFIG_FILE=benchy-run-${SLURM_JOBID}.yaml

export MASTER_ADDR=$(hostname)

set -x # enables debugging output, so each command that is run will be printed to the console.
# srun -u shifter -V ${DATADIR}:/data -V ${LOGDIR}:/logs \
srun \
    bash -c "
    source export_DDP_vars.sh
    ${PROFILE_CMD} python train_singleGPUdataLoader.py ${args}
    "


# rm benchy-run-${SLURM_JOBID}.yaml