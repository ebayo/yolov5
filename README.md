# YOLOv5 for Traffic Sign and Road Marking Detection

Forked from [Ultralytics/YOLOv5 v5.0](https://github.com/ultralytics/yolov5/releases/tag/v5.0)

Repository used in completion of the master thesis Traffic Sign Detection for Micromobility for [ETSETB](telecos.upc.edu/en) (Abstract below)

> In recent years there’s been an increase in the number or users of micromobility vehicles for everyday commute inside the city. At the same time technologies for autonomous and assisted driving for cars have been improving and entering the market. In this thesis we propose a proof of concept to join both fields and develop an automatic detector using a Convolutional Neural Network for road markings and traffic signs pertaining to micromobility using a relatively small network and a simple tracker we achieve good results. This projects also introduces a new database of traffic signs and road markings.

## Modifications of the original code

The code in this fork adds a custon data augmentation for training using the library [ImgAug](https://github.com/aleju/imgaug)

  - The file [data_augmentation.py](utils/data_augmentation.py) contains the definition of the augmentation used.
  - In [datasets.py](utils/datasets.py) the object is created when initialising the dataloader and used for training.
  - The files `data/hyp.**.DA*.yaml` contain the hyperparameters used for training with custom data augmentation.


## Experiments

Network trained for traffic sign and road marking detection used a custom database. See runs at [ts_detect W&B](https://wandb.ai/ebayo/ts_detect).


## Further work

See [ts_tracker](https://github.com/ebayo/ts_tracker) for the code used for tracking the detections of the network and other code used for analysing the results and working with the database.
