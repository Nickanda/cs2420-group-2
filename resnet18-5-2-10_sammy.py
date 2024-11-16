!pip install fvcore
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import AdamW
import time
from torchvision.models import resnet18
from fvcore.nn import FlopCountAnalysis, parameter_count_table

class Config:
    in_channels = 3
    num_classes = 10
    batch_size = 64
    lr = 1e-3
    epochs = 10
    num_students = 5
    hidden_dim = 256
    temperature = 3.0
    alpha = 0.7
    teacher_model_path = "teacher.pth"
    student_model_path = "student_{}.pth"

config = Config()

class Top1Gating(nn.Module):
    def __init__(self, dim, num_gates, eps=1e-9, outer_expert_dims=tuple(), capacity_factor_train=1.25, capacity_factor_eval=2.0):
        super().__init__()
        self.eps = eps
        self.num_gates = num_gates
        self.w_gating = nn.Parameter(torch.randn(dim, num_gates))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

    def forward(self, x, importance=None):
        b, d = x.shape
        num_gates = self.num_gates

        raw_gates = torch.matmul(x, self.w_gating)
        raw_gates = F.softmax(raw_gates, dim=-1)

        gate_1, index_1 = raw_gates.max(dim=-1)
        mask_1 = F.one_hot(index_1, num_gates).float()

        density_1 = mask_1.mean(dim=0)
        density_1_proxy = raw_gates.mean(dim=0)
        loss = (density_1_proxy * density_1).mean() * float(num_gates ** 2)

        return None, index_1, loss

def get_data_loaders(num_students=5, classes_per_group=None):
    if classes_per_group is None:
        classes_per_group = 10 // num_students
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    all_classes = list(range(10))

    class_groups = []
    for i in range(num_students):
        group_classes = all_classes[i * classes_per_group:(i + 1) * classes_per_group]
        class_groups.append(group_classes)
    
    def filter_by_classes(dataset, classes):
        indices = [i for i, target in enumerate(dataset.targets) if target in classes]
        return Subset(dataset, indices)

    train_loaders = []
    test_loaders = []
    for i, group in enumerate(class_groups):
        train_subset = filter_by_classes(train_dataset, group)
        test_subset = filter_by_classes(test_dataset, group)
        
        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_subset, batch_size=config.batch_size, shuffle=False, num_workers=2)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    
    return train_loader, test_loader, train_loaders

class TeacherModel(nn.Module):
    def __init__(self, num_classes=config.num_classes):
        super(TeacherModel, self).__init__()
        self.network = resnet18(pretrained=False)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

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

def train_epoch(model, loader, optimizer, criterion, device, description="Model"):
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
    
    print(f"{description} Train Loss: {total_loss / len(loader):.4f}, "
          f"Train Acc: {correct / len(loader.dataset):.4f}")
    return total_loss, correct / len(loader.dataset)

def evaluate_with_metrics(model, loader, device, description="Model"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct = 0, 0
    total_samples = 0
    
    with torch.no_grad():        
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            if i == 0:
                start_time = time.time()

                flops_input = inputs[:1].to(device)
                flops_analysis = FlopCountAnalysis(model, flops_input)
                flops_per_image = flops_analysis.total() / batch_size

                end_time = time.time()

                latency = (end_time - start_time) / batch_size

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
            total_samples += batch_size

    accuracy = correct / total_samples
    print(f"{description} Results:")
    print(f"Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.4f}")
    print(f"Latency per Image: {latency:.6f} secs")
    print(f"FLOPs per Image: {flops_per_image / 1e6:.2f} MFLOPs")

    return total_loss, accuracy, latency, flops_per_image

def distill_teacher_to_student(teacher, student, loader, optimizer, criterion, device):
    teacher.eval()
    student.train()
    total_loss = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_outputs = teacher(inputs)
            teacher_soft = F.softmax(teacher_outputs / config.temperature, dim=1)

        student_outputs = student(inputs)
        student_soft = F.log_softmax(student_outputs / config.temperature, dim=1)

        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (config.temperature ** 2)
        hard_loss = F.cross_entropy(student_outputs, targets)
        loss = config.alpha * distill_loss + (1 - config.alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Distill Loss: {total_loss / len(loader):.4f}")
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
        input_features = x.view(batch_size, -1)
        
        _, best_experts, _ = self.gating_net(input_features)

        outputs = torch.zeros(batch_size, self.students[0].network[-1].out_features).to(x.device)

        for i, expert_idx in enumerate(best_experts):
            outputs[i] = self.students[expert_idx.item()](x[i].unsqueeze(0)).squeeze(0)

        return outputs

def train_model(model, train_loader, criterion, optimizer, device, description, epochs=config.epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_epoch(model, train_loader, optimizer, criterion, device, description=description)

def train_moe_model(moe_model, loader, criterion, optimizer, device, epochs=1):
    moe_model.train()
    for epoch in range(epochs):
        total_loss, correct = 0, 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = moe_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets).sum().item()
        
        print(f"MoE Train Loss: {total_loss / len(loader):.4f}, "
              f"Train Acc: {correct / len(loader.dataset):.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader, train_loaders = get_data_loaders(num_students=5, classes_per_group=2)

    teacher = TeacherModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(teacher.parameters(), lr=config.lr)

    print("\nTeacher:")
    train_model(teacher, train_loader, criterion, optimizer, device, description="Teacher")

    torch.save(teacher.state_dict(), config.teacher_model_path)
    
    students = [StudentModel().to(device) for _ in range(config.num_students)]

    for i, student in enumerate(students):
        student_optimizer = AdamW(student.parameters(), lr=config.lr)
        print(f"\nDistilling Student {i+1}:")
        distill_teacher_to_student(teacher, student, train_loader, student_optimizer, criterion, device)

        print(f"\nTraining Student {i+1}:")
        train_model(student, train_loaders[i], criterion, student_optimizer, device, description=f"Student {i+1}")

        torch.save(student.state_dict(), config.student_model_path.format(i))

    gating_net = Top1Gating(dim=3 * 32 * 32, num_gates=config.num_students).to(device)
    moe_model = MoE(students, gating_net).to(device)

    print("\nTraining MoE Model:")
    moe_optimizer = AdamW(moe_model.parameters(), lr=config.lr)
    train_moe_model(moe_model, train_loader, criterion, moe_optimizer, device, epochs=config.epochs)

    print("\Teacher model:")
    evaluate_with_metrics(teacher, test_loader, device, description="Teacher")

    print("\MoE Model:")
    evaluate_with_metrics(moe_model, test_loader, device, description="MoE")

if __name__ == "__main__":
    main()
