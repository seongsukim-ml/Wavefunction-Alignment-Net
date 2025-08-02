python pipelines/train.py --config-name=config.yaml log_dir="$$AMLT_OUTPUT_DIR" \
wandb.open=True wandb.wandb_group="H" job_id="water-bz256" \
dataset_path="/dev/shm/water.db" ngpus=4 lr=0.0005 enable_hami=True \
hami_weight=1  batch_size=256 schedule=polynomial schedule.lr_warmup_steps=1000 \
max_steps=200000 used_cache=True \
train_ratio=500 val_ratio=500 gradient_clip_val=5.0 dataset_size=4999