

import torch
import torch.nn as nn
import torch.optim as optim

class CudaTestNet(nn.Module):

    def __init__(self):
        super(CudaTestNet, self).__init__()
        self.fc1 = nn.Linear(212, 256)
        self.fc2 = nn.Linear(256, 256)
        self.last_fc = nn.Linear(256, 5)

        self.stack = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.fc2,
            nn.ReLU(),
            self.last_fc,
            nn.ReLU()
        )
    
    def forward(self, obs):
        action = self.stack(obs)
        return action

def test(device):
    testNet = CudaTestNet().to(device)
    adam = optim.Adam(testNet.parameters(), lr=1e-3)

    for i in range(2):
        print("Epoch ", i+1, "!")
        obs = torch.rand((1028, 212)).to(device)
        actions = testNet(obs)
        
        target = torch.zeros((1028, 5)).to(device)
        criterion = nn.MSELoss()

        loss = criterion(actions, target)
        
        adam.zero_grad()
        loss.backward()
        adam.step()

        print(loss.detach().to(torch.device("cpu")).numpy())

if __name__ == "__main__":
    
    cpu = torch.device("cpu")
    print("Using device: ", cpu)
    test(cpu)

    cuda = torch.device("cuda")
    print("Using device: ", cuda)
    test(cuda)

    
