import torch
# import torch.nn as nn
from model1 import model1
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def setting():
	global CLASSES, device, train_loader, test_loader, trans
	CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
	torch.manual_seed(2018)
	usd_cuda = torch.cuda.is_available()
	device = torch.device('cuda')
	kwargs = {'num_workers':4, 'pin_memory':True}
	trans = transforms.Compose([
	    #transforms.Scale(32), # 32by32
	    transforms.ToTensor(),
	    transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
	])
	# train_loader = torch.utils.data.DataLoader(
	#     datasets.CIFAR10('./data', train = True, download = True,
	#                 transform = trans),
	#     batch_size = 256,
	#     shuffle=True,
	#     **kwargs)
	image_data = datasets.ImageFolder('./data/train/', transform=trans)
	train_loader = torch.utils.data.DataLoader(image_data,
	                                            batch_size=256,
	                                            shuffle = True,
	                                           num_workers = 8)
	test_loader = torch.utils.data.DataLoader(
	    datasets.CIFAR10('./data', train = False, download = False,
	                transform = trans),
	    batch_size = 256,
	    shuffle=True,
	    **kwargs)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 2560 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'. format(
                100.*batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

def kaggle():
	global kaggle_loader
	image_data = datasets.ImageFolder('./data/test/', transform=trans)
	kaggle_loader = torch.utils.data.DataLoader(image_data,
	                                            batch_size=256,
	                                            shuffle = False,
	                                           num_workers = 8)
if __name__ == '__main__':
	setting()
	# model = model1().to(device)
	model = torch.load("main1.pt").to(device)
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	kaggle()
	for i, (data, target) in enumerate(kaggle_loader):
		data = data.to(device)
		output = model(data)
		pred = output.max(1, keepdim = True)[1]
		for i, p in enumerate(pred):
			print(i+1, CLASSES[int(p)])
		break

	# for epoch in range(1,31):
	#     train(epoch)
	#     test()
	# else:
	#     torch.save(model, 'main1.pt')
