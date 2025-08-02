python ./pipelines/test_model.py --config-name=config.yaml \
       wandb.open=True wandb.wandb_group=Jia-test wandb.wandb_api_key=231fdd2c5f23b319b6fe54997e3454ec24684cf5 \
       log_dir=./local_files/logs \
       data_name=pubchem dataset_path=/scratch/data/pubchem_test \
       job_id=hami_test \
       basis=def2-tzvp \
       model_backbone=QHNet_backbone_symmetrySO2 \
       model.embedding_dimension=96 model.use_equi_norm=True model.max_radius=5 \
       output_model=EquivariantScalar_viaTP_Order0 \
       ngpus=1 \
       lr=0.001 enable_hami=True enable_energy=False enable_forces=False \
       hami_weight=2.39 energy_weight=0.01 forces_weight=0.6 batch_size=1 \
       schedule=polynomial schedule.lr_warmup_steps=1000 max_steps=300000 used_cache=True \
       train_ratio=0.9 val_ratio=0.06 test_ratio=0.04 \
       debug=False\
       test_energy_hami=True