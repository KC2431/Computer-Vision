import argparse
import json
import torch
from models import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.optim import Adam, Adagrad, SGD

if __name__ == "__main__":

    #-------------------------------- Reading in the command line arguments --------------------------------#
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', choices=['ResNet50', 'VGG19', 'ResNet20', 'WideResNet', 'BasicCNN'],
                        help=('Choices ResNet50 and VGG19 for ImageNet or NIPS2017 datasets.'
                              'Choices ResNet20, WideResNet and BasicCNN for CIFAR10 and CIFAR100 datasets.'
                              'BasicCNN for MNIST.'), type=str, required=True)
    parser.add_argument('--dataSet', choices=['CIFAR10', 'CIFAR100', 'NIPS2017', 'ImageNet', 'MNIST']
                        , type=str, required=True)
    parser.add_argument('--optim', choices=['SGD', 'Adam', 'Adagrad'], type=str, required=True)
    parser.add_argument('--maxIterations', type=int, required=True)
    parser.add_argument('--batchSize', type=int, required=True)

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    if args.dataSet in ['ImageNet', 'NIPS2017'] and args.model in ['ResNet20', 'WideResNet', 'BasicCNN']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataset}.")
    elif args.dataSet == 'CIFAR10' and args.model in ['ResNet50', 'VGG19']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataset}.")   
    elif args.dataSet == 'MNIST' and args.model in ['ResNet50', 'VGG19', 'ResNet20', 'WideResNet']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}") 
    #-------------------------------- Reading Data --------------------------------#
    
    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    ) 

    #-------------------------------- Intializing the Data Loader --------------------------------#

    batchSize=100
    dataLoader=DataLoader()