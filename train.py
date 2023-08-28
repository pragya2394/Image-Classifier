import matplotlib.pyplot as plt

import argparse
import torch
from collections import OrderedDict
from torch import nn, optim
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import seaborn as sb
import json

def argparser():
    parser = argparse.ArgumentParser (description = "train.py")

    parser.add_argument('--save_dir', action="store", dest="save_dir", type = str)
    parser.add_argument('--arch', dest="arch", action="store", default="vgg19", type = str)
    parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.01)
    parser.add_argument('--hidden_units', action="store", dest="hidden_units", default=512, type=int)
    parser.add_argument('--epochs', action="store", dest="epochs", default=20, type=int)
    parser.add_argument('--gpu', action="store", dest="gpu", default="gpu")
    arg = parser.parse_args()
    return arg

def validate(model, valid_loader, criterion):
    model.to (device)
    valid_loss, accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)

            valid_loss += criterion(output, labels).item()

            # Calculate accuracy
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss, accuracy

def main():
    args = argparser()

    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64, shuffle=True)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")


    model = models.vgg19(pretrained=True)
    for param in model.parameters():
            param.requires_grad = False 

    classifier = nn.Sequential(OrderedDict([('hidden1', nn.Linear(25088, 4096)),('relu1', nn.ReLU()),
                                ('dropout1', nn.Dropout(p = 0.3)),
                                ('hidden2', nn.Linear(4096, 1028)),('relu2', nn.ReLU()),
                                ('dropout2', nn.Dropout(p = 0.3)), 
                                ('hidden3', nn.Linear(1028, 102)),('relu2', nn.ReLU()),
                                ('output', nn.LogSoftmax(dim =1))
                                           ]))
    model.classifier = classifier

    criterion = nn.NLLLoss() # defining loss

    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001) # to update weights of network

    model.to(device) 
    epochs = 7
    print_every = 40
    steps = 0

    for epoch in range (epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss, accuracy = 0,0
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validate(model, valid_loader, criterion)
                print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                          f"Valid accuracy: {accuracy/len(validloader)*100:.3f}")
                running_loss = 0
                model.train()

    model.to ('cpu') 
    model.class_to_idx = train_datasets.class_to_idx 
    checkpoint = {'input_size':25088,
                    'output_size':102,
                    'hidden_layers':[each for each in model.classifier],
                     'droupout':0.3,
                     'epochs':7,
                     'classifier': model.classifier,
                     'state_dict':model.state_dict(),
                     'class_to_idx':model.class_to_idx}
    torch.save(checkpoint, 'checkpoint.pth')
    print('checkpoint has been saved!')
if __name__ == '__main__': 
    main()
