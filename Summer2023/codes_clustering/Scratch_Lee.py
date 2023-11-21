import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import time

# 모델 구조 불러오기
alexnet = torchvision.models.alexnet(pretrained=False)
alexnet.classifier[6] = nn.Linear(in_features = 4096, out_features = 10) # 데이터셋에 맞춰 output layer 변경

# 데이터 불러오기 (비교만 할거니까 전체 데이터 중 0.01 % 만 가져오자)
# Transformations for the CIFAR-10 dataset
transform = transforms.Compose(
    [transforms.Resize((224, 224)),  # Resize to 224x224
     transforms.RandomHorizontalFlip(),
     transforms.RandomCrop(224, padding=4),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


full_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Calculate the size of the subset (1/10 of the full dataset)
subset_size = len(full_trainset) // 100

# Use SubsetRandomSampler to get a subset of the data
subset_indices = list(range(subset_size))
subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(subset_indices)

trainloader = torch.utils.data.DataLoader(
    full_trainset, batch_size=256, sampler=subset_sampler,num_workers=0)
device_name = ['cuda:0', 'cpu']

device_time = {'cpu': [], 'cuda:0': []}
device_loss = {'cpu': [], 'cuda:0': []}

for i in range(2):
    device = device_name[i]
    model = alexnet.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    start_time = time.time()
    print('==' * 20)
    print(f'{device} 학습 시작')
    for epoch in range(10):

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        end_time = time.time()
        cumalative_time = end_time - start_time
        device_time[device].append(cumalative_time)
        device_loss[device].append(running_loss)
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)} Training Time : {cumalative_time}")
