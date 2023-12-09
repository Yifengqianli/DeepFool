
import os
import argparse
import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from model import ResNet18
from deepfool import deepfool
import model
from PIL import Image


def train(args, net, trainloader, testloader, pre_epoch, EPOCH, optimizer, loss_fn, writer, device, ):
    if not os.path.exists(args.outf):
        os.makedirs(args.outf)
    best_acc = 85  # 设置初始最佳准确率
    print("Start Training ResNet18")
    with open("acc.txt", "w") as f:
        with open("log.txt", "w") as f2:
            for epoch in range(pre_epoch, EPOCH):
                print("\nEpoch:%d" % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                for i, data in enumerate(trainloader, 0):
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()

                    outputs = net(inputs)
                    loss = loss_fn(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch"%d, itrer:%d] Loss:%.03f | Acc:%.3f%%'
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                    # writer.add_scalar("train_loss", loss.item(), i)
                    # writer.add_scalar("train_acc", 100. * correct / total, i)
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    writer.add_scalar("test_acc", acc, epoch + 1)
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
            return net


# 使用对抗样本攻击模型
def adver_attack_model(model, adver_example, target, name, batch_size, device, adver_nums, ):

    """print the correct number of the model and accuracy of the model.
       :param model: model to attack.
       :param adver_example: the adversarial example we use.
       :param target: the targets of examples.
       :param name: the name of model.
       :return:None
    """
    adver_dataset = TensorDataset(adver_example, target)

    loader = DataLoader(dataset=adver_dataset, batch_size=batch_size)
    correct_num = torch.tensor(0).to(device)
    for j, (images, targets) in tqdm(enumerate(loader)):
        images, targets = images.to(device), targets.to(device)
        pred = model.forward(images).max(1)[1]
        num = torch.sum(pred == targets)
        correct_num = correct_num + num
        print(correct_num)
    print(correct_num)
    print('\n{} on DeepFool Adversarial Example correct rate is {}'.format(name, correct_num/adver_nums))



# 可视化展示
def plot_clean_and_adver(adver_example, adver_target, clean_example, clean_target, ):
    n_cols = 5
    n_rows = 5
    cnt = 1
    cnt1 = 1
    plt.figure(figsize=(n_cols*4, n_rows*2))
    for i in range(n_cols):
        for j in range(n_cols):
            plt.subplot(n_cols, n_rows*2, cnt1)
            plt.xticks([])
            plt.yticks([])
            plt.title("{} -> {}".format(clean_target[cnt], adver_target[cnt]))
            plt.imshow(clean_example[cnt].permute(1, 2, 0).to("cpu").detach().numpy().astype('uint8'))
            plt.subplot(n_cols, n_rows*2, cnt1+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(adver_example[cnt].permute(1, 2, 0).to("cpu").detach().numpy().astype('uint8'))
            cnt = cnt + 1
            cnt1 = cnt1 + 2

    plt.show()


def main():
    # 超参数设置
    EPOCH = 135  # 遍历训练集次数
    pre_epoch = 0  # 定义已经遍历的训练集的次数
    BATCH_SIZE = 128  # 批处理尺寸
    LR = 0.01  # 学习率
    batch_size = 10  # 批处理尺寸
    adver_nums = 1000

    # 定义GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.ResNet18().to(device)
    adver_example_DeepFool = torch.zeros((batch_size, 3, 32, 32)).to(device)
    adver_target = torch.zeros(batch_size).to(device)
    clean_example = torch.zeros((batch_size, 3, 32, 32)).to(device)
    clean_target = torch.zeros(batch_size).to(device)

    # 参数设置
    parser = argparse.ArgumentParser(description="Pytorch CIFAR10 Training")
    parser.add_argument('--outf', default='./model/', help='folder to output source_images and model checkpoints')
    args = parser.parse_args()

    # 加载并预处理数据集
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 图像翻转
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 下载数据集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 数据集的标签类别
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    train_data_size = len(trainset)  # 记录数据集长度
    test_data_size = len(testset)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    writer = SummaryWriter("../logs_train")
    net = train(args, net, trainloader, testloader, pre_epoch, EPOCH, optimizer, loss_fn, writer, device=device, )

    for i, (images, targets) in enumerate(testloader):
        images, targets = images.to(device), targets.to(device)
        if i >= adver_nums/batch_size:
            break
        if i == 0:
            clean_example = images
        else:
            clean_example = torch.cat((clean_example, images), dim=0)

        cur_adver_example = torch.zeros_like(images).to(device)

        for j in range(batch_size):
            r_tot, loop_i, label, k_i, pert_image = deepfool(images[j], net)
            cur_adver_example[j] = pert_image

        pred = net(cur_adver_example).max(1)[1]

        if i == 0:
            adver_example_DeepFool = cur_adver_example
            clean_target = targets
            adver_target = pred

        else:
            adver_example_DeepFool = torch.cat((adver_example_DeepFool, cur_adver_example), dim=0)
            clean_target = torch.cat((clean_target, targets), dim=0)
            adver_target = torch.cat((adver_target, pred), dim=0)

    print(adver_example_DeepFool.shape)
    print(adver_target.shape)
    print(clean_example.shape)
    print(clean_target.shape)
    adver_attack_model(net, adver_example=adver_example_DeepFool, target=clean_target, name="ResNet18",
                       batch_size=batch_size, device=device, adver_nums=adver_nums)

    plot_clean_and_adver(adver_example=adver_example_DeepFool, adver_target=adver_target, clean_example=clean_example,
                         clean_target=clean_target)
    writer.close()

if __name__ == "__main__":
    main()