import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import generate_sexp_with_variable
import generate_dyck
import Recursive_Ttansformer
import transformer_dick_fixed_embed
import matrix_visualizer


def quick_train(model,device train_loader, epochs=1, lr=0.1):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))

    for ep in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

def train(device):
    # ===== 1) CIFAR-10 用データ前処理・DataLoader =====
    # 学習時の標準的なData Augmentation＋正規化（CIFAR-10の平均・分散）
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2470, 0.2435, 0.2616)),
    ])

    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                            std=(0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root='./data', train=True, download=True,
                                transform=transform_train)
    test_set  = datasets.CIFAR10(root='./data', train=False, download=True,
                                transform=transform_eval)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True,
                            num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=256, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = make_model().to(device)
    # 例：1epochだけ回す（時間節約のため任意）
    quick_train(model,device train_loader, epochs=1, lr=0.1)

    model.eval()
    errors=[]
    for xb, yb in test_loader:
        out=model(xb)
        error=out-yb
        errors.append(error)
    print("errors",errors)