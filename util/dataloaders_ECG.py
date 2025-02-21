import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from model.vit_autoencoder import vit_autoencoder
import numpy as np
import utils.lr_sched as lr_sched
import time
from dataset_args import ECGDataset
from sklearn.model_selection import train_test_split
from early_stopper import EarlyStopper
import matplotlib.pyplot as plt


# Step 1: Load and Preprocess ECG Data
ecg_data = np.load(r'L:\basic\diva1\Onderzoekers\DEEP RISK 2\ECGs DISTANT + DEEPRISK\ECGs.npy')
ecg_data = np.transpose(ecg_data, (0, 2, 1))  # (patients, leads, samples)
ecg_data = ecg_data[:, None, :, :]  # Add channel dimension (patients, 1, 12, 2500)

#Z-score Normalization
# Step 2: Remove signals that are all zeros
valid_indices = ~np.all(ecg_data == 0, axis=(1, 2, 3))
ecg_data = ecg_data[valid_indices]
print(f"Removed {len(valid_indices) - np.sum(valid_indices)} all-zero signals.")
print(f"Remaining signals for training: {len(ecg_data)}")

# Step 3: Normalize the Data
ecg_data = (ecg_data - np.mean(ecg_data)) / np.std(ecg_data)
print("Step 1 achieved: ECG data loaded and normalized")

# Step 2: Split Data into Train, Validation, and Test Sets
# Split into test (20%) and remaining (80%)
train_val_data, test_data = train_test_split(ecg_data, test_size=0.2, random_state=42)

# Split remaining into train (80% of 80%) and validation (20% of 80%)
train_data, val_data = train_test_split(train_val_data, test_size=0.2, random_state=42)

print(f"Train set size: {len(train_data)}, Validation set size: {len(val_data)}, Test set size: {len(test_data)}")

# Convert to Datasets
train_dataset = ECGDataset(train_data)
val_dataset = ECGDataset(val_data)
test_dataset = ECGDataset(test_data)

# Step 3: Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
print("Step 2 achieved: Train, validation, and test loaders created")

# Step 4: Initialize Model, Loss Function, and Optimizer
model = vit_autoencoder(img_size=(12, 2500), patch_size=(1, 100), in_chans=1)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
#%%
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize lists to store loss values
train_losses = []
val_losses = []

# Step 5: Training Loop with Validation and Early Stopping
num_epochs = 100
early_stopper = EarlyStopper(patience=5, min_delta=0.01)

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    for i, inputs in enumerate(train_loader):
        inputs = inputs.to(device)

        # Update progress display
        print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}", end='\r')

        # Forward pass
        outputs, _ = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)  # Store training loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs.to(device)
            outputs, _ = model(inputs)
            val_loss += criterion(outputs, inputs).item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  # Store validation loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

    # Save model if validation loss improves
    if avg_val_loss < early_stopper.min_validation_loss:
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Model saved at epoch {epoch+1} with validation loss {avg_val_loss:.4f}.")

    # Debug EarlyStopping behavior
    print(f"Epoch [{epoch+1}/{num_epochs}] EarlyStopping Counter: {early_stopper.counter}, Min Validation Loss: {early_stopper.min_validation_loss:.4f}")

    # Early stopping check
    if early_stopper.early_stop(avg_val_loss):
        break

print("Training complete.")

# Plotting Training and Validation Losses
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
#%%


def load_and_evaluate(train_loader,test_loader, model_path):
    # Step 1: Load the trained model
    model_path = model_path
    model = vit_autoencoder(img_size=(12, 2500), patch_size=(1, 100), in_chans=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")

    # Step 2: Evaluate the model on the test set
    # Assuming `test_loader` is already created as in your training script
    test_loss = 0.0
    criterion = torch.nn.MSELoss()
    inputs_list, outputs_list,latents_list = [], [],[]

    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)  # Extract the tensor from the dataset tuple
            outputs, latents = model(inputs)

            # Compute loss
            loss = criterion(outputs, inputs)
            test_loss += loss.item()

            # Save inputs and outputs for plotting
            inputs_list.append(inputs.cpu())
            outputs_list.append(outputs.cpu())
            latents_list.append(latents.cpu())

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    # Step 3: Plot input vs output ECG signals
    plot_input_output(inputs_list, outputs_list)
    return inputs_list, outputs_list, latents_list


def plot_input_output(inputs_list, outputs_list, num_samples=5):
    """
    Plot input ECG signals and their corresponding output signals.
    :param inputs_list: List of input tensors from the test set
    :param outputs_list: List of output tensors from the model
    :param num_samples: Number of samples to plot
    """
    inputs = torch.cat(inputs_list)  # Combine batches
    outputs = torch.cat(outputs_list)  # Combine batches

    # Ensure the number of samples doesn't exceed the dataset size
    num_samples = min(num_samples, inputs.size(0))

    for i in range(num_samples):
        input_signal = inputs[i, 0, 3, :].numpy()  # Extract lead 0
        output_signal = outputs[i, 0, 3, :].numpy()  # Extract lead 0

        plt.figure(figsize=(10, 4))

        # Plot input signal on the left
        plt.subplot(1, 2, 1)
        plt.plot(input_signal, label="Input ECG")
        plt.title(f"Sample {i+1} - Input Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot output signal on the right
        plt.subplot(1, 2, 2)
        plt.plot(output_signal, label="Output ECG", color="orange")
        plt.title(f"Sample {i+1} - Output Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()

        plt.tight_layout()
        plt.show()

inputs, outputs, latents = load_and_evaluate(train_loader, test_loader, 'best_model.pth')


def plot_12_leads(inputs_list, outputs_list, patient_idx=0):
    """
    Plot the 12-lead ECG input signals and their corresponding output signals for a single patient.
    :param inputs_list: List of input tensors from the test set
    :param outputs_list: List of output tensors from the model
    :param patient_idx: Index of the patient to plot
    """
    # Combine batches into a single tensor
    inputs = torch.cat(inputs_list)
    outputs = torch.cat(outputs_list)
   
    # Extract input and output for the specified patient
    input_signals = inputs[patient_idx].squeeze(0).numpy()  # Shape: (1, 12, Time)
    output_signals = outputs[patient_idx].squeeze(0).numpy()  # Shape: (12, 2500)
       
    num_leads = input_signals.shape[0]
   
    # Create a (3, 4) grid for the 12 leads
    fig, axes = plt.subplots(3, 4, figsize=(15, 10),sharex=True, sharey=True)
    fig.suptitle(f"12-Lead ECG Comparison for Patient {patient_idx + 1}")
   
    lead_names = [
        "Lead I", "Lead II", "Lead III",
        "aVR", "aVL", "aVF",
        "V1", "V2", "V3",
        "V4", "V5", "V6"
    ]
   
    for lead_idx in range(num_leads):
        row, col = divmod(lead_idx, 4)
        ax = axes[row, col]
       
        # Extract signals for the current lead
        input_signal = input_signals[lead_idx, :]
        output_signal = output_signals[lead_idx, :]
       
        # Plot the input and output signals
        ax.plot(input_signal, label="Input ECG", color="blue")
        ax.plot(output_signal, label="Output ECG", color="orange", alpha=0.7)
        ax.set_title(lead_names[lead_idx])
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
   
    # Adjust layout to avoid overlapping titles and labels
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


plot_12_leads(inputs, outputs, patient_idx=0)