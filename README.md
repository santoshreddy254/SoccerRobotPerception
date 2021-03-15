#  Soccer Robot Perception

## Team Members:
### Jaswanth Bandlamudi
### Venkata Santosh Sai Ramireddy Muthireddy


## Structure
model.py : NimbroNet2 architecture is defined here

dataloader.py : Dataloader for loading detection and segmentation data

losses.py : Losses for detection and segmentation dataset are defined here

metrics.py : metrics for measuring the performance of the detection and segmentation models

train.py : A script to train the NimbroNet2 model

### To Train NimbroNet2 model:
python main.py --batch_size BATCH_SIZE --num_epochs NUM_EPOCHS --dataset_path PATH_TO_DATA --save_path PATH_TO_SAVE


## TO DO:

Code Documentation

Linting

PEP8 check