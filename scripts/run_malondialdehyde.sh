python pipelines/train.py --config-name=config.yaml \
wandb.open=True wandb.wandb_group="H" job_id="m-bz256" \
dataset_path="/dev/shm/malondialdehyde.db" ngpus=4 lr=0.0005 enable_hami=True \
hami_weight=1  batch_size=32 schedule=polynomial schedule.lr_warmup_steps=1000 \
max_steps=200000 used_cache=True \
train_ratio=25000 val_ratio=500 gradient_clip_val=5.0 dataset-size=26978