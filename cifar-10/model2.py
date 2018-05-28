import torch.nn as nn
import torch.nn.functional as F
#model
class model2(nn.Module):
    def __init__(self):
        super(model2,self).__init__()
        self.conv1 = nn.Conv2d(3,9,kernel_size=3)
        self.conv2 = nn.Conv2d(9,9, kernel_size=3)
        self.conv3 = nn.Conv2d(9,18, kernel_size=3)
        self.conv4 = nn.Conv2d(18,18, kernel_size=3)

        # self.conv5 = nn.Conv2d(18,36, kernel_size=3)

        self.fc1 = nn.Linear(450,450)
        self.fc2 = nn.Linear(450,450)
        self.fc3 = nn.Linear(450,10)

    def forward(self, x):
        x = F.relu(self.conv1(x)) #32 -> 30
        x = F.relu(F.max_pool2d(self.conv2(x),2)) #30 -> 28 -> 14
        x = F.relu(self.conv3(x)) #14 -> 12
        x = F.relu(F.max_pool2d(self.conv4(x),2)) # 12 -> 10 -> 5
        # x = F.relu(self.conv5(x)) # 5 -> 3
        
        x = x.view(-1,450)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)