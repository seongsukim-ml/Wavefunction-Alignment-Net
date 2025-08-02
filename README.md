# Introduction
This repository contains our latest code for DFT acceleration with Neural Network. 
Generally you should put all your local configs and runnig parameters in `local_files` folder. This folder is ignored by git.
# Setup environment
## Local environment
By running the following command, you will create a virtual environment with all the necessary packages to run the code.
Note that it is better to use CUDA driver more than 520, due to the requirement of the package `CUDA Toolkit 12`.
```bash
# Create a model training environment.
conda env create -n madft_nn -f environment.yaml
# install the latest amlt, this is for submit job
conda create -n amlt9 python=3.10
pip install amlt==9.23 --extra-index-url https://msrpypi.azurewebsites.net/stable/leloojoo
```
## Using Docker
You can also use the docker image in your local machine. 
You can mount any folder to this docker container via -v LOCAL_PATH:/mnt
```bash
az login --use-device-code
az acr login -n msrmoldyn
docker pull msrmoldyn.azurecr.io/feynman:madft-nn-py311torch210-py310torch120
nvidia-docker run -it --gpus all --ipc=host -v /:/mnt b7ccf18b577a /bin/bash
```


# Mount the data
The data is stored in a Azure container. To mount the data, you should have access to the container: [ai4science0eastus](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/3eaeebff-de6e-4e20-9473-24de9ca067dc/resourceGroups/electronic-structure-rg/providers/Microsoft.Storage/storageAccounts/ai4science0eastus/overview). If you have no access to this container, please contact the Jia Zhang.

1. Create configuration file as follows and save it as `local_files/teamdrive.cfg`
    ```
    accountName ai4science0eastus
    accountKey YourKey
    sasToken YourToken
    containerName madft-nn
    ```
    You can get the `accountKey` and `sasToken` from the Azure portal: [Key](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/3eaeebff-de6e-4e20-9473-24de9ca067dc/resourceGroups/electronic-structure-rg/providers/Microsoft.Storage/storageAccounts/ai4science0eastus/keys), [Token](https://ms.portal.azure.com/#@microsoft.onmicrosoft.com/resource/subscriptions/3eaeebff-de6e-4e20-9473-24de9ca067dc/resourceGroups/electronic-structure-rg/providers/Microsoft.Storage/storageAccounts/ai4science0eastus/sas). Make sure that no `" "` used for the key and token.
2. Login in Azure
    ```bash
    az login --use-device-code
    ```
3. Run the following command to mount the data
    ```bash
    # It is better to set up a folder in /tmp and pass the absolute path when mounting the data
    sudo mkdir /tmp/tmp_teamdrive sudo chmod -R 777 /tmp/tmp_teamdrive

    blobfuse ../mnt --tmp-path=/tmp/tmp_teamdrive --config-file=local_files/teamdrive.cfg -o attr_timeout=240 -o entry_timeout=240 -o negative_timeout=120
    ```
    You can install blob by running `sudo apt install blobfuse`. It is better to set `data` as the mount point to align with the config.
    How to use these data please refer to [here](./notebook/lmdb_data_demo.ipynb) for `lmdb` and [here](./notebook/db_data_demo.ipynp) for `db` file.

# Model training
## Halmintonian model train on local machine

1. Run the following command to train the model
    ```bash
     python pipelines/train.py --wandb=True --wandb-group="test" --dataset-path="/dev/shm/QH9_new.db" \
    --ngpus=4 --lr=0.0005 --enable-hami=True --hami-weight=1  --batch-size=32 \
    --lr-warmup-steps=1000 --lr-schedule="polynomiallr" --max-steps=300000 --used-cache=True \
    --train-ratio=0.9 --val-ratio=0.06 --test-ratio=0.04  --gradient-clip-val=5.0 --dataset-size=100000
    ```
## Halmintonian model train on Azure ML
1. Add A100 cluster
    ```bash
    # add our a100 cluster (if you want to run on azure cluster, this step is necessary, else, skip is ok.)
    amlt target add --service aml --workspace-name electronic-structure-ws --resource-group electronic-structure-ws-rg --subscription 3eaeebff-de6e-4e20-9473-24de9ca067dc
    amlt target list aml
    # you can checkout to this project
    # you can also init a new project by yourself.
    amlt project checkout MADFT-NN qcacceleration madft-nn-data
    ```
2. Submit job
    ```bash
    cd amltyaml
    amlt run run_nc96.yaml
    ```
3. Model debug on cluster
    On amlk8s cluster (FYR https://amulet-docs.azurewebsites.net/main/advanced/2_testing.html) To ssh into a running job, use  

    ```
    amlt ssh <experiment_name> [:<job_name>]
    ```
### Azure cluster

aml

amlk8s

sing

a100

## LSRM model for energy and force prediction
---------------------------------------------------

# Data construction by MADFT
```
cd amltyaml
# the cluster is 16g v100 low priority, you can prepare data with these machine
amlt target add ai4s-es-madft-v100lowpri --service aml --cluster v100 --workspace-name moldynws --resource-group shared_infrastructure --subscription "Molecular Dynamics"
export NUM_WORKERS=[int] # how many workers for each moldataset
# please set the required dataset_input_dir in run_data_onsingularity.yaml
# if workers = 8, and dataset_input_dir has 4 dataset, 32 v100-G1 task will start.
amlt run run_data_onsingularity.yaml
```

please notice when calling data_gen_pubchem.py in azure: 
--output_dir "$$AMLT_DATA_DIR/../pubchem_20231117/"
You must mkdir this folder in your azure storage first!!!!!!
e.g. dataset/filtered_data is the remote dir. and "dataset/pubchem_20231117/" is the output dir.

---------------------------------------------

# Reference code repos:

```
git clone https://github.com/shehzaidi/pre-training-via-denoising.git
cd pre-training-via-denoising```
