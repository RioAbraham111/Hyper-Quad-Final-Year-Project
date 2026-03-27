import torch.nn as nn
import torch
import numpy as np

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
    
    def fit(self, X, y, loss_function, optimizer, nTrainSteps):
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
    
    def predict(self, theta_data, theta_dot_data, tau_data, X_mean, X_std):
        sim_data = np.stack([theta_data, theta_dot_data, tau_data], axis=1)
        X_new = torch.tensor(sim_data, dtype=torch.float32)
        X_new = X_new.reshape(1, -1)
        X_new = (X_new - X_mean) / X_std
        self.eval()
        with torch.no_grad():
            y_pred = self.forward(X_new)

        return y_pred

    