import torch.nn as nn
import torch.nn.functional as F
#model
class model1(nn.Module):
    def __init__(self):
        super(model1,self).__init__()
        self.conv1 = nn.Conv2d(3,10,kernel_size=3)
        self.conv2 = nn.Conv2d(10,20, kernel_size=3)
        self.conv3 = nn.Conv2d(20,40, kernel_size=3)
#         self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1440,840)
        self.fc2 = nn.Linear(840,160)
        self.fc3 = nn.Linear(160,10)

    def forward(self, x):
        x = F.relu(self.conv1(x),2)
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3(x),2))
        
        x = x.view(-1,1440)
        
        x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)