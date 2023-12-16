# Anti-Spoofing (RawNet2) project

 This is a repository for Anti-Spoofing project based on one homework of DLA course (HSE). The task is to determine whether the audio is real (human) or created using special algorithms (TTS, VC, mix).
## Repository structure

`hw_as` - directory included all project files (only what was used in this project is indicated).
* `base` - base classes for model and train.
* `collate_fn` - function for collation.
* `datasets` - functions and class for parsing protocols from ASVSpoof2019 (Dataset), audio preprocessing (we need 4 sec during both train and evaluation part).
* `configs` - configs with params for training.
* `logger` - files for logging.
* `loss` - definition for loss computation (Weighted-CrossEntropy).
* `metric` - Equal Error Rate (from [HSE DLA Course, week10](https://github.com/XuMuK1/dla2023/blob/2023/hw5_as/calculate_eer.py)).
* `model` - RawNet2 model architecture (from [END-TO-END ANTI-SPOOFING WITH RAWNET2](https://arxiv.org/pdf/2011.01108.pdf)).
* `test_data` - this folder contains 7 examples of audio files for test.
* `trainer` - train loop, logging in W&B.
* `utils` - crucial functions (parse_config, object_loading, utils).

## Installation guide

As usual, clone repository, change directory and install requirements:

```shell
!git clone https://github.com/KemmerEdition/hw_as.git
!cd /content/hw_as
!pip install -r requirements.txt
```
## Train
Train model with command below (you should use Kaggle for training because there is a dataset needed in this task. You need to add the [dataset](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) to your workspace, only then run the script).
   ```shell
   !python -m train -c hw_5/configs/rawnet2_train.json
   ```
If you want to resume training from checkpoint, use command below.
   ```shell
   !python -m train -c hw_5/configs/rawnet2_train.json -r checkpoint-epoch15-0.pth
   ```
## Test 
You only need to run following commands (download checkpoint of my model, run test.py), wait some time and enjoy.

You should test this on Google Colab (GPU to prevent any problems with tensors on different devices) or the path to the test audio directory should be changed in test.py (for test you should use second one).
   ```shell
#   Epoch 15
   !gdown --id 19JL64GGojQJmn5QDU-y5eObtT6hh7rNA
#   Epoch 100
   !gdown --id 1mbUiJdaIkyLcDnAwLHLtNJvhaGLoX99z
  ```
   ```shell
!python -m test -c hw_5/configs/rawnet2_train.json -r checkpoint-epoch100.pth 
   ```
Enjoy!
## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
