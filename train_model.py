import torch
from tqdm import tqdm

class TrainModel():
    def __init__(self, 
                 model,
                 maxIters, 
                 device, 
                 trainDataloader, 
                 testDataloader,
                 lossFunction,
                 optim,
                 isANN=False,
                 saveModel=False,
                 ver=False):
     
        self.model = model
        self.maxIters = maxIters
        self.device = device
        self.trainDataloader = trainDataloader
        self.testDataloader = testDataloader
        self.saveModel = saveModel
        self.ver = ver
        self.lossFunction = lossFunction
        self.optim = optim
        self.isANN = isANN

    def train(self, args):
        self.model = self.model.to(self.device)

        for epoch in range(self.maxIters):

            trainingLoop = tqdm(iterable=enumerate(self.trainDataloader), 
                            leave=True,
                            total=len(self.trainDataloader))

            for _, (x, y) in trainingLoop:

                x = x.to(self.device)
                y = y.to(self.device)

                if self.isANN:
                    x = x.reshape(x.shape[0], -1)

                pred = self.model(x)
                loss = self.lossFunction(pred, y)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                with torch.no_grad():
                    mask = (torch.argmax(pred, dim=1) == y)
                    acc = mask.float().mean()

                trainingLoop.set_description(f"[Epoch {epoch+1}/{self.maxIters}]")
                trainingLoop.set_postfix(loss=loss.item(), acc=acc.item())

        if self.saveModel:
            torch.save(self.model.state_dict(), f'Trained_Models/{args.model}_{args.dataSet}_trained_model.pt')    
