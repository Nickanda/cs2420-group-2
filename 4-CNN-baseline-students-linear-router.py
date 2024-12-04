# -*- coding: utf-8 -*-
"""Copy of PostInterim_MoIST_120224.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TcJ69xuojU0T5oH-la7XAvabBLu3jB3F
"""

!pip install fvcore
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
from transformers import AdamW
from torchvision.models import resnet18
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import defaultdict

class Config:
    in_channels = 3
    num_classes = 10
    batch_size = 64
    lr = 1e-3
    epochs = 15
    num_students = 4
    hidden_dim = 256
    temperature = 3.0
    alpha = 0.7
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
    def __init__(self, num_classes=config.num_classes):
        super(TeacherModel, self).__init__()
        self.network = resnet18(pretrained=True)
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

class GatingNetwork(nn.Module):
    def __init__(self, num_students, input_dim):
        super(GatingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_students)
        )
        self.temperature = 5.0  # High initial temperature for exploration

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits / self.temperature, dim=1)  # Apply temperature scaling


class MoE(nn.Module):
    def __init__(self, students, gating_net):
        super(MoE, self).__init__()
        self.students = nn.ModuleList(students)
        self.gating_net = gating_net

    def forward(self, x, return_router_assignments=False):
        batch_size = x.size(0)
        gating_probs = self.gating_net(x.view(batch_size, -1))  # Router probabilities
        best_experts = gating_probs.argmax(dim=1)  # Selected experts for each input

        outputs = torch.zeros(batch_size, self.students[0].network[-1].out_features).to(x.device)
        for i, expert_idx in enumerate(best_experts):
            outputs[i] = self.students[expert_idx](x[i].unsqueeze(0)).squeeze(0)

        if return_router_assignments:
            return outputs, best_experts
        return outputs
        
def distill_teacher_to_student(teacher, student, loader, optimizer, criterion, device):
    teacher.eval()
    student.train()
    total_loss = 0
    num_correct = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            teacher_outputs = teacher(inputs)
            teacher_soft = F.softmax(teacher_outputs / config.temperature, dim=1)

        student_outputs = student(inputs)
        num_correct += (student_outputs.argmax(1) == targets).sum().item()
        student_soft = F.log_softmax(student_outputs / config.temperature, dim=1)

        distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (config.temperature ** 2)
        hard_loss = F.cross_entropy(student_outputs, targets)
        loss = config.alpha * distill_loss + (1 - config.alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Distill Loss: {total_loss / len(loader):.4f}")
    print(f"Distill Acc: {num_correct / len(loader):.4f}")
    return total_loss / len(loader)

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

def visualize_specialization(class_prob_tracker, num_classes, num_students):
    # Convert class probabilities to a NumPy array for easy plotting
    data = torch.stack([class_prob_tracker[c] for c in range(num_classes)]).cpu().numpy()

    # Set up bar width and positions
    bar_width = 0.75  # Make bars wider for better visualization
    x_indices = np.arange(num_classes)

    # Define unique colors for each student
    student_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot bars for each student
    bottom_values = np.zeros(num_classes)  # Initialize bottoms for stacked bars
    for student_id in range(num_students):
        student_probs = data[:, student_id]  # Probabilities for this student
        ax.bar(
            x_indices,
            student_probs,
            bar_width,
            bottom=bottom_values,
            color=student_colors[student_id],
            label=f"Student {student_id}",
            alpha=0.9
        )
        bottom_values += student_probs  # Update bottom for stacking

    # Customize the plot
    ax.set_xlabel("Classes")
    ax.set_ylabel("Router Probability")
    ax.set_title("Class Specialization Across Students")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"Class {i}" for i in range(num_classes)])
    ax.legend(title="Students", loc="upper right")

    plt.tight_layout()
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get CIFAR-10 data loaders
    train_loader, test_loader = get_data_loaders()

    # Load Teacher Model
    teacher = TeacherModel().to(device)
    teacher.load_state_dict(torch.load("teacher.pth", map_location=device))
    teacher.eval()  # Ensure the teacher is in evaluation mode
    print("Teacher model loaded successfully.")

    # Distillation Phase: Teacher -> Students
    print("\nDistilling Teacher Knowledge into Students:")
    students = [StudentModel().to(device) for _ in range(config.num_students)]
    for i, student in enumerate(students):
        optimizer_student = AdamW(student.parameters(), lr=config.lr)
        for epoch in range(config.epochs // 2):  # Train for half of total epochs
            distill_teacher_to_student(teacher, student, train_loader, optimizer_student, nn.CrossEntropyLoss(), device)
        torch.save(student.state_dict(), config.student_model_path.format(i))

    # Mixture of Experts Training
    print("\nTraining MoE Model:")
    gating_net = GatingNetwork(num_students=config.num_students, input_dim=3 * 32 * 32).to(device)
    moe_model = MoE(students, gating_net).to(device)

    optimizer_moe = AdamW(list(gating_net.parameters()) + [p for student in students for p in student.parameters()], lr=config.lr*2)
    class_prob_tracker = defaultdict(lambda: torch.zeros(config.num_students, device=device))  # Tracker for specialization

    for epoch in range(config.epochs):
      print(f"Epoch {epoch+1}/{config.epochs}")
      moe_model.train()

      total_loss, correct = 0, 0
      for inputs, targets in train_loader:
          inputs, targets = inputs.to(device), targets.to(device)

          optimizer_moe.zero_grad()

          # Forward pass
          gating_probs = moe_model.gating_net(inputs.view(inputs.size(0), -1))
          outputs = moe_model(inputs)

          # Standard classification loss
          classification_loss = nn.CrossEntropyLoss()(outputs, targets)

          # Diversity loss: Encourage balanced assignments early
          diversity_loss = -torch.mean(torch.sum(gating_probs * torch.log(gating_probs + 1e-8), dim=1))

          # Specialization loss: Encourage monopolization per class
          with torch.no_grad():
              for class_id in range(config.num_classes):
                  class_indices = (targets == class_id)
                  if class_indices.sum() > 0:
                      class_prob_tracker[class_id] += gating_probs[class_indices].mean(dim=0)
          specialization_loss = 0
          for class_id in range(config.num_classes):
              class_probs = class_prob_tracker[class_id] / class_prob_tracker[class_id].sum()
              specialization_loss += -torch.log(torch.max(class_probs) + 1e-8)  # Penalize small contributions
          specialization_loss /= config.num_classes

          # Entropy loss: Prevent overconfidence early
          entropy_loss = -torch.mean(torch.sum(gating_probs * torch.log(gating_probs + 1e-8), dim=1))

          # Underutilization loss: Penalize students that are not being used
          utilization = gating_probs.mean(dim=0)
          underutilization_loss = torch.sum((1.0 / config.num_students - utilization) ** 2)

          # Combined loss
          #loss = classification_loss + 0.1 * diversity_loss + 20.0 * specialization_loss + 0.1 * entropy_loss + 0.5 * underutilization_loss
          #loss = classification_loss + 0.1 * diversity_loss + 50.0 * specialization_loss + 0.1 * entropy_loss + 1.0 * underutilization_loss
          loss = classification_loss + 0.1 * diversity_loss + 100.0 * specialization_loss + 0.1 * entropy_loss + 1.0 * underutilization_loss

          # Penalize uniform router probabilities
          router_diversity_loss = torch.sum((gating_probs.mean(dim=0) - 1.0 / config.num_students) ** 2)
          loss += 0.5 * router_diversity_loss

          loss.backward()
          optimizer_moe.step()

          total_loss += loss.item()
          correct += (outputs.argmax(1) == targets).sum().item()

      accuracy = correct / len(train_loader.dataset)
      print(f"MoE Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {accuracy:.4f}")

      # Anneal temperature
      gating_net.temperature = max(0.01, gating_net.temperature * 0.7)  # Faster annealing


    # Evaluate and visualize specialization
    print("\nEvaluating MoE Model:")
    evaluate_with_metrics(moe_model, test_loader, device, description="MoE")
    visualize_specialization(class_prob_tracker, config.num_classes, config.num_students)


if __name__ == "__main__":
    main()
