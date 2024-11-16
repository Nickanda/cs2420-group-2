# code taken from:
# https://github.com/lucidrains/mixture-of-experts/blob/master/mixture_of_experts/mixture_of_experts.py
# https://github.com/ShivamRajSharma/Teacher-Student-Network/blob/main/train.py

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import AdamW
from sklearn.model_selection import train_test_split
from typing import List

class Config:
    in_channels = 3
    num_classes = 10
    batch_size = 64
    lr = 1e-3
    epochs = 10
    num_students = 4
    hidden_dim = 256
    teacher_model_path = "teacher.pth"
    student_model_path = "student_{}.pth"

config = Config()

def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 128, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x):
        return self.network(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(8 * 8 * 64, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x):
        return self.network(x)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

def distill_teacher_to_student(teacher, student, loader, optimizer, criterion, temperature, alpha, device):
    teacher.eval()
    student.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_outputs = teacher(inputs)
            teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)

        student_outputs = student(inputs)
        student_soft = F.log_softmax(student_outputs / temperature, dim=1)

        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
        hard_loss = F.cross_entropy(student_outputs, targets)
        loss = alpha * distill_loss + (1 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

class GatingNetwork(nn.Module):
    def __init__(self, num_students, input_dim):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_students)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=1)

class MoE(nn.Module):
    def __init__(self, students, gating_net):
        super(MoE, self).__init__()
        self.students = nn.ModuleList(students)
        self.gating_net = gating_net

    def forward(self, x):
        batch_size = x.size(0)
        gating_probs = self.gating_net(x.view(batch_size, -1))
        outputs = torch.stack([student(x) for student in self.students], dim=1)
        moe_output = (outputs * gating_probs.unsqueeze(-1)).sum(dim=1)
        return moe_output

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_data_loaders()
    teacher = TeacherModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(teacher.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        train_loss, train_acc = train_epoch(teacher, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(teacher, test_loader, criterion, device)
        print(f"Teacher Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    torch.save(teacher.state_dict(), config.teacher_model_path)

    students = [StudentModel().to(device) for _ in range(config.num_students)]
    for i, student in enumerate(students):
        student_optimizer = AdamW(student.parameters(), lr=config.lr)
        for epoch in range(config.epochs):
            distill_loss = distill_teacher_to_student(teacher, student, train_loader, student_optimizer, criterion, temperature=3.0, alpha=0.7, device=device)
            print(f"Student {i + 1}, Epoch {epoch + 1}: Distill Loss: {distill_loss:.4f}")

        torch.save(student.state_dict(), config.student_model_path.format(i))

    gating_net = GatingNetwork(num_students=config.num_students, input_dim=3 * 32 * 32).to(device)
    moe_model = MoE(students, gating_net).to(device)

    val_loss, val_acc = evaluate(moe_model, test_loader, criterion, device)
    print(f"MoE Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    main()
