# Computer Vision
- This repository contains code for training various deep learning models (like `VGG19`, `ResNet50`, `ResNet20`, etc.) for various datasets.

### Setup
- Dependencies: `pandas`, `torch`, `torchvision`, `natsort`, `pillow`, `tqdm`

- The NIPS2017 data set can be downloaded from [here](https://www.kaggle.com/competitions/nips-2017-defense-against-adversarial-attack/data). To train a model, execute the main file with numerous arguments. For e.g., `python main.py --model 'ResNet20' --dataSet 'CIFAR10' --optim 'Adam' --maxIterations 10 --batchSize 64 --numWorkers 2 --save
Model 1` will train a `ResNet20` model on the `CIFAR10` data set using the [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) optimizer with a batch size of 64 along with max epochs of 10. The `--numworkers` option allows the user to select the number of workers for loading the dataset. Set `--saveModel` to 1 to save the model otherwisee set to 0.
