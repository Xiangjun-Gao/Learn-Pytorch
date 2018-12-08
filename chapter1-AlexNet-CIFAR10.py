import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import time
from tensorboardX import SummaryWriter

# 超参数设置
EPOCH = 50   #遍历数据集次数
BATCH_SIZE = 4      #批处理尺寸(batch_size)
LR = 0.001        #学习率

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='/media/mcislab/GaoXiangjun/Learn_Pytorch/data/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root='/media/mcislab/GaoXiangjun/Learn_Pytorch/data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model


def main():

    writer = SummaryWriter('AlexNet_CIFAR')
    net =alexnet()
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

    for epoch in range(EPOCH):  # loop over the dataset multiple times
        t0 = time.time()
        running_loss = 0.0
        total_train = 0
        correct_train = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            _, predicted = torch.max(outputs.detach(), 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # 计算loss    反向传播
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1,
                                                running_loss / 2000))
                writer.add_scalar("train_loss", running_loss/2000, (epoch)*len(trainloader)/2000 + i/2000)
                running_loss = 0.0
                t1 = time.time()
                #print('epoch:%d     batch:%d    time per 2000 batches:%lf' %
                #      (epoch+1, i+1, t1 - t0))
                t0 = time.time()

        print('epoch:%d    train_acc：%d%%' % (epoch + 1, (100 * correct_train / total_train)))
        writer.add_scalar('train_acc', (100 * correct_train / total_train),epoch+1)

        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images = images.cuda()
                labels = labels.cuda()
                outputs = net(images)
                _, predicted = torch.max(outputs.detach(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('epoch:%d    val_acc：%d%%' % (epoch + 1, (100 * correct / total)))
            writer.add_scalar('val_acc', (100 * correct / total),epoch + 1)

        # 保存每个epoch下的模型参数
        torch.save(net.state_dict(), './AlexNet_CIFAR/AlexNet_CIFAR_%03d_params.pkl' % (epoch + 1))

    print('Finished Training')
    writer.export_scalars_to_json("./AlexNet_CIFAR/AlexNet_CIFAR.json")
    writer.close()

    # 5. Test the network on the whole test data
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.detach(), 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print('Accuracy of the network on the 10000 test images: %d %%' %
                  (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.detach(), 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main()
