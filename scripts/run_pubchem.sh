python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group="Lin-pubchem" wandb.wandb_api_key=21dcfbbb833c4ab9c4ccb098bd70e3298bddb800 \
data_name="pubchem" dataset_path="/data/used_data/pubchem_20230831_processed/data.0000.lmdb" basis="def2-tzvp"  model.embedding_dimension=96  model.max_radius=5 output_model=EquivariantScalar_viaTP_Order0 \
ngpus=4 lr=0.001 enable_hami=True hami_weight=1 batch_size=8 schedule=polynomial \
schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True \
train_ratio=0.9 val_ratio=0.06 test_ratio=0.04 gradient_clip_val=5.0 energy_train_loss=huber forces_train_loss=huber hami_train_loss=huber remove_init=True remove_atomref_energy=True

python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group="Lin-pubchem" wandb.wandb_api_key=21dcfbbb833c4ab9c4ccb098bd70e3298bddb800 \
data_name="pubchem" dataset_path="/data/used_data/pubchem_20230831_processed/data.0000.lmdb" basis="def2-tzvp"  model.embedding_dimension=96  model.max_radius=5 output_model=EquivariantScalar_viaTP_Order0 \
ngpus=4 lr=0.001 enable_energy=True  enable_forces=True energy_weight=0.01 forces_weight=0.99  batch_size=8 schedule=polynomial \
schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True \
train_ratio=0.9 val_ratio=0.06 test_ratio=0.04 gradient_clip_val=5.0 energy_train_loss=huber forces_train_loss=huber hami_train_loss=huber remove_init=True remove_atomref_energy=True


python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group="Lin-pubchem" wandb.wandb_api_key=21dcfbbb833c4ab9c4ccb098bd70e3298bddb800 \
data_name="pubchem" dataset_path="/data/used_data/pubchem_20230831_processed/data.0000.lmdb" basis="def2-tzvp"  model.embedding_dimension=96  model.max_radius=5 output_model=EquivariantScalar_viaTP_Order0 \
ngpus=4 lr=0.001 enable_hami=True hami_weight=2.39 enable_energy=True  enable_forces=True energy_weight=0.01 forces_weight=0.6  batch_size=8 schedule=polynomial \
schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True \
train_ratio=0.9 val_ratio=0.06 test_ratio=0.04 gradient_clip_val=5.0 energy_train_loss=huber forces_train_loss=huber hami_train_loss=huber remove_init=True remove_atomref_energy=True

python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group="Lin-pubchem" wandb.wandb_api_key=231fdd2c5f23b319b6fe54997e3454ec24684cf5 \
log_dir="$AMLT_OUTPUT_DIR" data_name="pubchem" dataset_path="/tmp/dataset" job_id=fix_pubchemALL_ef_symso2_eqnorm \
basis="def2-tzvp" model_backbone=QHNet_backbone_symmetrySO2 model.embedding_dimension=96 model.use_equi_norm=True model.max_radius=5 output_model=EquivariantScalar_viaTP_Order0 \
ngpus=4 lr=0.001 enable_energy=True enable_forces=True hami_weight=2.39 energy_weight=0.01 forces_weight=0.6  batch_size=8 schedule=polynomial \
schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True