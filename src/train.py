import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# 1. Load Original Data

X_sig = np.load('X2_data.npy').astype('float32')
X_feat = np.load('X2_features.npy').astype('float32')
y = np.load('Y2_labels.npy').astype('int')

# 2. Combine Signals + Features
X_combined = np.hstack([
    X_sig.reshape(len(X_sig), -1),
    X_feat
])


# 3. Train / Validation / Test Split
# 70% Train - 30% Temp
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    X_combined,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 15% Validation - 15% Test
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# 4. Apply SMOTE on Training Set Only
sm = SMOTE(random_state=42)

X_train_res, y_train = sm.fit_resample(
    X_train_raw,
    y_train_raw
)

# 5. Separate Signals and Features Again
def process_back(data):

    sigs = data[:, :120].reshape(-1, 1, 120)  # 1 channel
    feats = data[:, 120:]

    return sigs, feats

X_train_sig, X_train_feat = process_back(X_train_res)
X_val_sig, X_val_feat = process_back(X_val_raw)
X_test_sig, X_test_feat = process_back(X_test_raw)


# 6. Smooth Only New Synthetic Training Samples
indices_new = np.arange(
    len(X_train_raw),
    len(X_train_res)
)

for i in indices_new:

    X_train_sig[i, 0, :] = np.convolve(
        X_train_sig[i, 0, :],
        np.ones(5) / 5,
        mode='same'
    )

# 7. Combine Signals + Features for Temporal Model
X_train_final = np.concatenate(
    [X_train_sig, X_train_feat[:, np.newaxis, :]],
    axis=2
)

X_val_final = np.concatenate(
    [X_val_sig, X_val_feat[:, np.newaxis, :]],
    axis=2
)

X_test_final = np.concatenate(
    [X_test_sig, X_test_feat[:, np.newaxis, :]],
    axis=2
)

print(
    f"Done! "
    f"Train: {len(X_train_final)}, "
    f"Val: {len(X_val_final)}, "
    f"Test: {len(X_test_final)}"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Make sure TemporalSleepModel is already defined
model.to(device)

# 9. Training Configuration

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0005,
    weight_decay=1e-4
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5
)

criterion = nn.CrossEntropyLoss()

best_val_acc = 0.0
for epoch in range(20):

    model.train()

    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        # Backpropagation
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total_train += batch_y.size(0)

        correct_train += (
            predicted == batch_y
        ).sum().item()

    train_acc = correct_train / total_train

    # VALIDATION 
    model.eval()

    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)

            _, predicted = torch.max(outputs.data, 1)

            total_val += batch_y.size(0)

            correct_val += (
                predicted == batch_y
            ).sum().item()

    val_acc = correct_val / total_val
    
    # Update learning rate
    scheduler.step(val_acc)

    # Save Best Model
    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            'best_temporal_model.pth'
        )

        msg = " Best Model Saved!"

    else:
        msg = ""

    #  Results
  

    print(
        f"Epoch [{epoch+1}/{epochs}] | "
        f"Train Acc: {100 * train_acc:.2f}% | "
        f"Val Acc: {100 * val_acc:.2f}% | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} "
        f"{msg}"
    )

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc * 100:.2f}%"
)

