import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(MLP, self).__init__()
        self.nInput  = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput
        self.linear1 = nn.Linear(self.nInput, self.nHidden)
        self.linear2 = nn.Linear(self.nHidden, self.nHidden)
        self.linear3 = nn.Linear(self.nHidden, self.nHidden)
        self.linear4 = nn.Linear(self.nHidden, self.nOutput)
        self.ReLU    = nn.ReLU()

    def forward(self, x):
        h1 = self.ReLU(self.linear1(x))
        h2 = self.ReLU(self.linear2(h1))
        h3 = self.ReLU(self.linear3(h2))
        output = self.linear4(h3)
        return(output)
    
    def train(self, X, y, loss_function, optimizer, nTrainSteps):
        lossses = []
        for epoch in range(nTrainSteps):
            current_loss = 0.0
            for xBatch, yBatch in [(X, y)]:
                optimizer.zero_grad() 
                output = self.forward(xBatch) 
                loss = loss_function(output, yBatch)
                lossses.append(loss.item())
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
            if (epoch + 1) % 100 == 0: 
                print(f'Epoch [{epoch + 1}/{nTrainSteps}], Loss: {current_loss / len([(X, y)]):.4f}')
        
        return lossses

    