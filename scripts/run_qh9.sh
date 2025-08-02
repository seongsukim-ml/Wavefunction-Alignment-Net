python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group="Lin" job_id=QHNet_No wandb.wandb_api_key=21dcfbbb833c4ab9c4ccb098bd70e3298bddb800 \
dataset_path="/data/used_data/QH9_new.db" model_backbone=QHNet_backbone_No ngpus=4 lr=0.0005 enable_hami=True \
hami_weight=1  batch_size=32 schedule=polynomial schedule.lr_warmup_steps=1000 \
max_steps=300000 used_cache=True \
train_ratio=0.9 val_ratio=0.06 test_ratio=0.04  gradient_clip_val=5.0 dataset_size=100000


# python pipelines/train.py --config-name=config.yaml wandb.open=True wandb.wandb_group="QH9" job_id=QHNet_SO2 wandb.wandb_api_key=6f1080f993d5d7ad6103e69ef57dd9291f1bf366 \
# dataset_path="/gpfs/gibbs/pi/gerstein/yl2428/qh9/QH9_new.db" model_backbone=QHNetBackBoneSO2 ngpus=4 lr=0.0005 enable_hami=True \
# hami_weight=1  batch_size=32 schedule=polynomial schedule.lr_warmup_steps=1000 \
# max_steps=300000 used_cache=True \
# train_ratio=0.9 val_ratio=0.06 test_ratio=0.04  gradient_clip_val=5.0 dataset_size=100000