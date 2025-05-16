#!/bin/bash

#SBATCH -p gpu                  # Partition (queue)
#SBATCH -t 28:00:00            # Time limit hrs:min:sec
#SBATCH --gpus=1               # Number of GPUs
#SBATCH --constraint=a40      # Request specific GPU type
#SBATCH --mem=55G              # Memory per node
#SBATCH -o slurm-%j.out        # Standard output log
#SBATCH -e slurm-%j.err        # Standard error log
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80  # Email notification
#SBATCH --job-name=changeformer_train # Job name

module load conda/latest
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6
conda activate changeformer_new
cd ~/deforestation_detection/changeformer/ChangeFormer/
#GPUs
gpus=0,1

#Set paths
checkpoint_root=checkpoints
vis_root=vis
data_name=deforestation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
img_size=256    
batch_size=16   
lr=0.0001         
max_epochs=2
embed_dim=256

net_G=ChangeFormerV6        #ChangeFormerV6 is the finalized verion

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights
pretrain=None

#Train and Validation splits
split=train         #trainval
split_val=test      #test
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}

python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}
