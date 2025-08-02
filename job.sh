#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:4 
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

eval "$(conda shell.bash hook)"
conda activate madft_nn

# python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group=QH9_hyper_search \
# dataset_path=/gpfs/gibbs/pi/gerstein/yl2428/qh9/QH9_new.db ngpus=4 lr=0.0006 enable_hami=True hami_weight=1 \
# batch_size=32 schedule=polynomial schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True train_ratio=0.9 \
# val_ratio=0.06 test_ratio=0.04 gradient_clip_val=5.0 dataset_size=100000 ema_decay=0.995 multi_para_group=True \
# wandb.wandb_api_key=231fdd2c5f23b319b6fe54997e3454ec24684cf5 precision=32 model_backbone=QHNet_backbone_symmetrySO2 model=qhnet_so2

# python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group=QH9_model_dev \
# dataset_path=/gpfs/gibbs/pi/gerstein/yl2428/qh9/QH9_new.db ngpus=4 lr=0.0006 enable_hami=True hami_weight=1 \
# batch_size=32 schedule=polynomial schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True train_ratio=0.9 \
# val_ratio=0.06 test_ratio=0.04 gradient_clip_val=5.0 dataset_size=100000 ema_decay=0.995 multi_para_group=True \
# wandb.wandb_api_key=231fdd2c5f23b319b6fe54997e3454ec24684cf5 precision=32 model_backbone=TorchMD_Norm_Hami model=visnet job_id=visnet_first_run

python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_api_key=231fdd2c5f23b319b6fe54997e3454ec24684cf5 wandb.wandb_group=YL_Pubchem_Orbital_Energy dataset_path=/gpfs/gibbs/pi/gerstein/yl2428/pubchem_new data_name=pubchem basis=def2-tzvp ngpus=4 dataset_size=-1 lr=0.001 batch_size=8 enable_hami=True hami_weight=2.39 energy_weight=0.01 forces_weight=0.6 schedule=polynomial schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True remove_init=True model=equiformerv2 model_backbone=Equiformerv2SO2 model.order=3 model.max_radius=5 output_model=EquivariantScalar_viaTP enable_symmetry=True hami_model.name=HamiHeadSymmetry_uuw job_id=equiformerv2_hamiuuw_orbital_energy_weight_sweep_0_1 enable_hami_orbital_energy=True orbital_energy_weight=1 enable_energy_hami_error=False