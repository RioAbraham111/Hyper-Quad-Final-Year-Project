import torch as nn

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
    
    def train(self, train_dataloader, loss_function, optimizer, nTrainSteps):
        for epoch in range(nTrainSteps):
            current_loss = 0.0
            for xBatch, yBatch in train_dataloader:
                optimizer.zero_grad()  # Zero the gradients
                output = self.forward(xBatch)  # Forward pass
                loss = loss_function(output, yBatch)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                current_loss += loss.item()  # Accumulate loss for reporting
            if (epoch + 1) % 100 == 0:  # Print loss every 100 epochs
                print(f'Epoch [{epoch + 1}/{nTrainSteps}], Loss: {current_loss / len(train_dataloader):.4f}')

    