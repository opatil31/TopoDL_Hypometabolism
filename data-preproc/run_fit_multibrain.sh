#!/bin/bash
### Job name
#SBATCH -J Run_Fit_MultiBrain_Template
#SBATCH -o /home/hice1/khom9/CS8903/data-preproc/fit_multibrain_template.out
#SBATCH -e /home/hice1/khom9/CS8903/data-preproc/fit_multibrain_template.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=khom9@gatech.edu
### Queue name
### Specify the number of nodes and thread (ppn) for your job.
#SBATCH -N1 --ntasks=32
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1 -C HX00
### Tell PBS the anticipated run-time for your job, where walltime=HH:MM:SS
#SBATCH -t 16:00:00
#################################

cd $SLURM_SUBMIT_DIR
mkdir /tmp/multibrain-mri
module load matlab
# matlab -batch "fit_to_multibrain_job"
matlab -batch "fit_to_multibrain_template_job"

cd /tmp
mkdir multibrain-mri-y
# mkdir mri-template
mv multibrain-mri/y_* multibrain-mri-y
# mv multibrain-mri/*.mat multibrain-mri/softmax* multibrain-mri/mu* mri-template

# mv mri-template/ /home/hice1/khom9/scratch/
# echo "Copied template"
mv multibrain-mri-y/ /home/hice1/khom9/scratch/
echo "Copied files"
tar --use-compress-program="pigz" -cf multibrain-mri-y.tar.gz multibrain-mri-y/
echo "Compressed folder"
cp multibrain-mri-y.tar.gz /home/hice1/khom9/scratch/
echo "Done"