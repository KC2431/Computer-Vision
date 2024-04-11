import argparse
import json
import torch
import torch.utils
from models import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam, Adagrad, SGD
from utils import *

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
    parser.add_argument('--numWorkers',type=int, required=True)

    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_gpu else "cpu")

    if args.dataSet in ['ImageNet', 'NIPS2017'] and args.model in ['ResNet20', 'WideResNet', 'BasicCNN']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}.")
    elif args.dataSet == 'CIFAR10' and args.model in ['ResNet50', 'VGG19']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}.")   
    elif args.dataSet == 'MNIST' and args.model in ['ResNet50', 'VGG19', 'ResNet20', 'WideResNet']:
        raise ValueError(f"Can't use model {args.model} for dataset {args.dataSet}")

    #-------------------------------- Model Transformations --------------------------------#
    
    # Here, the first value is the training transformation and the second one is the test transformation
    modelTrainingTransforms = {

        'VGG19': [ 
                    transforms.Compose([
                           transforms.Resize(224),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(224, padding=10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                       ]) , 
                    
                    transforms.Compose([
                           transforms.Resize(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                       ])
                    ],
        
        'ResNet50': [
                        transforms.Compose([
                           transforms.Resize(224),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(224, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
                       ]),

                        transforms.Compose([
                           transforms.Resize(224),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                std=[0.229, 0.224, 0.225])
                       ])
                        ],
        
        'ResNet20':  [
                            transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, 4),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])
                            ]),

                            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])  
                            ])

                        ],   

        'WideResNet': [     
                            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])  
                            ]),

                            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])  
                            ]) 
                        ],
        
        'BasicCNN': [     
                            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])  
                            ]),

                            transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])  
                            ]) 
                        ] 
    
    }
    
    #-------------------------------- Initialising the dataset and dataloader --------------------------------#
    
    if args.dataSet == "MNIST":

        trainingData = datasets.MNIST(
            root="Data",
            train=True,
            download=True,
            transform=modelTrainingTransforms[args.model][0]
        )

        testData = datasets.MNIST(
            root="Data",
            train=False,
            download=True,
            transform=modelTrainingTransforms[args.model][1]
        ) 

        trainLoader = DataLoader(dataset=trainingData,
                                 batch_size=args.batchSize,
                                 shuffle=True,
                                 num_workers=args.numWorkers)
        
        testLoader = DataLoader(dataset=testData,
                                batch=args.batchSize,
                                shuffle=False,
                                num_workers=args.numWorkers)

    elif args.dataSet == 'CIFAR10':

        trainingData = datasets.CIFAR10(
            root="Data",
            train=True,
            download=True,
            transform=modelTrainingTransforms[args.model][0]
        )

        testData = datasets.CIFAR10(
            root="Data",
            train=False,
            download=True,
            transform=modelTrainingTransforms[args.model][1]
        )
        
    elif args.dataSet == 'CIFAR100':

        trainingData = datasets.CIFAR100(
            root="Data",
            train=True,
            download=True,
            transform=modelTrainingTransforms[args.model][0]
        )

        testData = datasets.CIFAR100(
            root="Data",
            train=False,
            download=True,
            transform=modelTrainingTransforms[args.model][1]
        )

    elif args.dataSet == 'NIPS2017':
        
        dataset = CustomDataset(img_dir='Data/neurips2017/images',
                             csv_file='Data/neurips2017/images.csv',
                             transform=transforms.ToTensor()
                             )
        
        trainingData, testData = torch.utils.data.random_split(dataset,[800, 200])
        
        trainDataTransform = modelTrainingTransforms[args.model][0]
        trainingData = trainDataTransform(trainingData)

        testDataTransform = modelTrainingTransforms[args.model][1]
        testData = testDataTransform(testData)

        
    trainLoader = DataLoader(dataset=trainingData,
                                 batch_size=args.batchSize,
                                 shuffle=True,
                                 num_workers=args.numWorkers)
        
    testLoader = DataLoader(dataset=testData,
                                batch=args.batchSize,
                                shuffle=False,
                                num_workers=args.numWorkers)
        
    
