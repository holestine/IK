from __future__ import print_function
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from vit_pytorch import ViT
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from timer import Timer
import numpy as np

LOG_DIR = "./logs/tb"

DEFAULT_EPOCHS = 15

DATASETS    = ["MNIST"] 
MODEL_TYPES = ["CNN"]
#DATASETS    = ["MNIST", "CIFAR10", "CIFAR100"] 
#MODEL_TYPES = ["MLP", "CNN", "ViT", "HYBRID"]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class CNNNet(nn.Module):
    def __init__(self, num_classes, channels, image_height, image_width):
        super(CNNNet, self).__init__()
        self.dropout = nn.Dropout(0.25)
        
        self.norm = nn.BatchNorm2d(32)
        
        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        fc_size = self.forward(torch.rand((1, channels, image_height, image_width)), True)

        self.fc1 = nn.Linear(fc_size, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, get_fc_size=False):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)

        if get_fc_size:
            return x.nelement()

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output

class MLPNet(nn.Module):
    def __init__(self, image_size, num_classes, channels):
        super(MLPNet, self).__init__()
        
        self.fc1 = nn.Linear(image_size * channels, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)
        return output  

class ViTNet(nn.Module): 
    def __init__(self, image_size, patch_size, num_classes, channels):
        HIDDEN_SIZE = 768
        DEPTH = 6
        HEADS = 12
        MLP_DIM = 3072
        
        super(ViTNet, self).__init__()
        
        self.model = ViT(
                        image_size = image_size,
                        patch_size = patch_size,
                        num_classes = num_classes,
                        dim = HIDDEN_SIZE,
                        depth = DEPTH,
                        heads = HEADS,
                        mlp_dim = MLP_DIM,
                        channels = channels,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )

    def forward(self, x):
        x = self.model(x)
        output = F.log_softmax(x, dim=1)
        return output

class HybridNet(nn.Module):
    def __init__(self, image_size, channels, num_classes):
        HIDDEN_SIZE = 768
        DEPTH = 6
        HEADS = 12
        MLP_DIM = 3072

        super(HybridNet, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        image_width = int((math.sqrt(image_size) - 4) / 2)

        self.vit = ViT(
                        image_size = image_width ** 2,
                        patch_size = int(image_width / 2),
                        num_classes = num_classes,
                        dim = HIDDEN_SIZE,
                        depth = DEPTH,
                        heads = HEADS,
                        mlp_dim = MLP_DIM,
                        channels = 64,
                        dropout = 0.1,
                        emb_dropout = 0.1
                        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.vit(x)

        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader, dataset, modeltype, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            with Timer("{} {}".format(modeltype, dataset)):
                output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    writer.add_scalar('Accuracy/{} {}'.format(modeltype, dataset), accuracy, epoch)
    return accuracy

def main(args):

    # Determine if we should use a GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    # Reduce precision to speed up training
    torch.set_float32_matmul_precision('medium')

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs  = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    accuracies = {}
    parameters = {}
    
    for dataset in DATASETS:
        if dataset == "MNIST":
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081))
            ])
            train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_set = datasets.MNIST('./data', train=False, transform=transform)
            image_height = 28
            image_width = 28
            patch_size = 14
            num_classes = 10
            channels = 1
        elif dataset == "CIFAR10":
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_set = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
            test_set = datasets.CIFAR10('./data', train=False, transform=transform)
            image_height = 32
            image_width = 32
            patch_size = 16
            num_classes = 10
            channels = 3
        elif dataset == "CIFAR100":
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_set = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
            test_set = datasets.CIFAR100('./data', train=False, transform=transform)
            image_height = 32
            image_width = 32
            patch_size = 16
            num_classes = 100
            channels = 3

        for model_type in MODEL_TYPES:
            print("Evaluating {} with {}".format(model_type, dataset))

            train_loader = torch.utils.data.DataLoader(train_set,**train_kwargs)
            test_loader  = torch.utils.data.DataLoader(test_set, **test_kwargs)

            if model_type == "MLP":
                model = MLPNet(image_height * image_width, num_classes, channels).to(device)
            elif model_type == "CNN":
                model = CNNNet(num_classes, channels, image_height, image_width).to(device)
            elif model_type == "ViT":
                model = ViTNet(image_height * image_width, patch_size, num_classes, channels).to(device)
            elif model_type == "HYBRID":
                model = HybridNet(image_height * image_width, channels, num_classes).to(device)
            
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

            max_accuracy = 0
            writer = SummaryWriter(log_dir=LOG_DIR)
            for epoch in range(1, args.epochs + 1):
                train(args, model, device, train_loader, optimizer, epoch)
                accuracy = test(model, device, test_loader, dataset, model_type, writer, epoch)
                max_accuracy = max(accuracy, max_accuracy)
                scheduler.step()
                if (max_accuracy > 98):
                    print('Breaking at epoch {}'.format(epoch))
                    break

            accuracies["{} {}".format(model_type, dataset)] = max_accuracy
            parameters["{} {}".format(model_type, dataset)] = count_parameters(model)

            if args.save_model:
                torch.save(model.state_dict(), "{}_{}.pt".format(model_type, dataset))

    #Timer().report_phases()

    times = Timer().get_phases()
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    bar_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:orange']
    ax1.bar(list(times.keys()), list(times.values()), color=bar_colors)
    ax1.set_ylabel('Inference Times (ms)')
    ax2.bar(list(times.keys()), list(parameters.values()), color=bar_colors)
    ax2.set_ylabel('Total Parameters')
    ax3.bar(list(times.keys()), list(accuracies.values()), color=bar_colors)
    ax3.set_ylabel('Model Accuracy')
    plt.show()
    plt.savefig('model_graph.jpg')
    print('DONE')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Transformer Experiments')
    parser.add_argument('--batch-size',      type=int,            default=64,             help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs',          type=int,            default=DEFAULT_EPOCHS, help='Number of epochs to train (default: 15)')
    parser.add_argument('--lr',              type=float,          default=.00001,         help='Learning rate (default: .00001)')
    parser.add_argument('--gamma',           type=float,          default=0.7,            help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed',            type=int,            default=1,              help='Random seed (default: 1)')
    parser.add_argument('--test-batch-size', type=int,            default=1000,           help='Input batch size for testing (default: 1000)')
    parser.add_argument('--log-interval',    type=int,            default=10,             help='How many batches to wait before logging training status (default: 10)')
    parser.add_argument('--dry-run',         action='store_true', default=False,          help='Quickly check a single pass (default: False)')
    parser.add_argument('--no-cuda',         action='store_true', default=False,          help='Disables CUDA training (default: False)')
    parser.add_argument('--save-model',      action='store_true', default=False,          help='For Saving the current Model (default: False)')
    
    args = parser.parse_args()

    main(args)
