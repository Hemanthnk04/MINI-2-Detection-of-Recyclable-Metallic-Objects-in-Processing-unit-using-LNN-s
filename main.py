import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import numpy as np

# Define the liquid state machine model with increased complexity
class LiquidStateMachine(nn.Module):
    def __init__(self, input_size, reservoir_size, output_size):
        super(LiquidStateMachine, self).__init__()
        self.reservoir_size = reservoir_size
        
        # Define the input to reservoir connection
        self.input_to_reservoir = nn.Linear(input_size, reservoir_size)
        
        # Define the reservoir to reservoir recurrent connections
        self.reservoir = nn.Linear(reservoir_size, reservoir_size)
        
        # Define the readout layer
        self.readout = nn.Linear(reservoir_size, output_size)
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        # Initialize the hidden state for the reservoir
        h = torch.zeros(batch_size, self.reservoir_size).to(x.device)
        
        for t in range(seq_length):
            # Input to reservoir transformation
            u = self.input_to_reservoir(x[:, t, :])
            
            # Reservoir state update with dropout
            h = F.dropout(F.relu(u + self.reservoir(h)), p=0.5, training=self.training)
        
        # Readout layer
        out = self.readout(h)
        return F.log_softmax(out, dim=1)
    
    def extract_features(self, x):
        batch_size, seq_length, _ = x.size()
        h = torch.zeros(batch_size, self.reservoir_size).to(x.device)
        for t in range(seq_length):
            u = self.input_to_reservoir(x[:, t, :])
            h = F.relu(u + self.reservoir(h))
        return h

# Define image augmentation with increased input size
input_image_size = (512, 512)
transform = transforms.Compose([
    transforms.Resize(input_image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Initialize the model, loss function, and optimizer
input_size = 512*512*3
reservoir_size = 1024
output_size = 9
model = LiquidStateMachine(input_size, reservoir_size, output_size)
criterion = nn.CrossEntropyLoss()

# Optimizer choices
optimizer_choices = {
    1: optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1),
    2: optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1),
    3: optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.1)
}
trainset = datasets.ImageFolder('split_dataset/train', transform=transform)
valset = datasets.ImageFolder('split_dataset/validation', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

def train_model():
    model_file_val = 'best_val_model.pt'
    model_file_train = 'best_train_model.pt'
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    
    # Choose optimizer
    optimizer_type = int(input("Press 1 for SGD, 2 for Adam, 3 for RMSprop: "))
    optimizer = optimizer_choices[optimizer_type]
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    # Early stopping parameters
    patience = 20
    wait = 0

    # Train the model
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(int(input("Enter number of epochs:"))):  # loop over the dataset multiple times
        model.train()
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs.view(-1, 1, input_size))  # flatten the input images and add time dimension
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
        train_accuracy = correct / len(trainset)
        train_loss = running_loss / len(trainloader)
        train_losses.append(train_loss)
        train_accs.append(train_accuracy * 100)
        print('Epoch %d, Training Loss: %.3f, Training Accuracy: %.2f%%' % (epoch + 1, train_loss, train_accuracy * 100))

        # Validate the model
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data in valloader:
                images, labels = data
                outputs = model(images.view(-1, 1, input_size))  # flatten the input images and add time dimension
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / len(valset)
        val_accs.append(val_accuracy * 100)
        val_loss /= len(valloader)
        val_losses.append(val_loss)
        print('Epoch %d, Validation Loss: %.3f, Validation Accuracy: %.2f%%' % (epoch + 1, val_loss, val_accuracy * 100))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_file_val)
            wait = 0  # reset patience counter
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            torch.save(model.state_dict(), model_file_train)
        
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break
        
        # Step the scheduler
        scheduler.step(val_loss)

    plt.figure(figsize=(12, 4))
    # Plot the training and validation losses
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(val_losses)), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_accs)), train_accs, label='Training Accuracy')
    plt.plot(range(len(val_accs)), val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.legend()
    plt.show()

def test_model():
    testset = datasets.ImageFolder('split_dataset/test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    # Choose which model to load
    model_choice = int(input("Press 1 to load best validation model, 2 to load best training model: "))
    if model_choice == 1:
        model_file = 'best_val_model.pt'
    elif model_choice == 2:
        model_file = 'best_train_model.pt'
    
    model = LiquidStateMachine(input_size, reservoir_size, output_size)
    model.load_state_dict(torch.load(model_file))

    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images.view(-1, 1, input_size))  # flatten the input images and add time dimension
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_accuracy = correct / len(testset)
    print('Test Loss: %.3f, Test Accuracy: %.2f%%' % (test_loss / len(testloader), test_accuracy * 100))

    # Plot confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=testset.classes, yticklabels=testset.classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    print("Classification Report:\n", classification_report(y_true, y_pred, target_names=testset.classes))

    # Extract features using the trained model
    def extract_features_from_loader(model, loader, device):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for data in loader:
                inputs, target_labels = data
                inputs = inputs.view(-1, 1, input_size).to(device)
                extracted_features = model.extract_features(inputs)
                features.append(extracted_features.cpu().numpy())
                labels.append(target_labels.numpy())
        features = np.concatenate(features)
        labels = np.concatenate(labels)
        return features, labels

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract features from training, validation, and test sets
    train_features, train_labels = extract_features_from_loader(model, trainloader, device)
    val_features, val_labels = extract_features_from_loader(model, valloader, device)
    test_features, test_labels = extract_features_from_loader(model, testloader, device)

    # Standardize features and train SVM
    svm = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1))
    svm.fit(train_features, train_labels)

    # Evaluate SVM on validation set
    val_predictions = svm.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print('SVM Validation Accuracy: {:.2f}%'.format(val_accuracy * 100))
    print('SVM Validation Classification Report:\n', classification_report(val_labels, val_predictions, target_names=valset.classes))

    # Evaluate SVM on test set
    test_predictions = svm.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print('SVM Test Accuracy: {:.2f}%'.format(test_accuracy * 100))
    print('SVM Test Classification Report:\n', classification_report(test_labels, test_predictions, target_names=testset.classes))
    
    # Plot confusion matrix for SVM on test set
    cm_svm = confusion_matrix(test_labels, test_predictions)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=testset.classes, yticklabels=testset.classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

if __name__ == "__main__":
    mode = int(input("Enter 1 to train the model, 2 to test the model: "))
    if mode == 1:
        train_model()
    elif mode == 2:
        test_model()
