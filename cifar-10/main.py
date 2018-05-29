import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os

def setting():
    global CLASSES, device, train_loader, test_loader, trans, kwargs, batch_size
    # CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    torch.manual_seed(2018)
    usd_cuda = torch.cuda.is_available()
    device = torch.device('cuda')
    kwargs = {'num_workers':8, 'pin_memory':True}
    batch_size = 4096
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    image_data = datasets.ImageFolder('./data/train/', transform=trans)
    CLASSES = image_data.classes
    train_loader = torch.utils.data.DataLoader(image_data,
                                                batch_size=batch_size,
                                                shuffle = True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train = False, download = True,
                    transform = trans),
        batch_size = batch_size,
        shuffle=True,
        **kwargs)

def train(epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'. format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100.*batch_idx / len(train_loader), loss.item()))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim = True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def kaggle_data():
    global kaggle_loader
    class MyImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            return super(MyImageFolder, self).__getitem__(index), self.imgs[index]
    image_data = MyImageFolder('./data/test/', transform=trans)
    kaggle_loader = torch.utils.data.DataLoader(image_data,
                                                batch_size=1,
                                                shuffle = False,
                                                ) # if setting num_worker -> error...fxxx...


if __name__ == '__main__':
    setting()

    # from model1 import model1
    # from model2 import model2
    from vgg16 import VGG

    model = VGG("VGG16")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    save_model_path = './model/'
    learning_rate = 0.001
    save_model_name = 'vgg16' + str(learning_rate) + '.pt'
    
    if save_model_name in os.listdir(save_model_path):
        print("make kaggle csv file")
        model = torch.load(save_model_path+save_model_name).to(device)
        kaggle_data()

        import csv
        csv_name = save_model_name[:-3] + '.csv'
        f = open('./rst/'+csv_name, 'w', newline = '')
        wr = csv.writer(f)
        wr.writerow(["id", "label"])
        cnt = 0
        for i, data in enumerate(kaggle_loader):
            if i % 1000 == 0:
                print(i)
            (data,_), (path,_) = data
            data = data.to(device)
            path = path[0][17:-4]
            output = model(data)
            pred = output.max(1, keepdim = True)[1]
            for i, p in enumerate(pred):
                wr.writerow([path, CLASSES[int(p)]])
            cnt += 1
        f.close()
    else:
        print("model training & save")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(1,251):
            train(epoch)
            if epoch % 5 == 0:
                test()
            if epoch % 100 == 0:
                torch.save(model, save_model_path+"epoch"+str(epoch)+save_model_name)
        else:
            torch.save(model, save_model_path+save_model_name)