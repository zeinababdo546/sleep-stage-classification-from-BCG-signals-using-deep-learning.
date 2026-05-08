# Load original data (before balancing)
X_sig = np.load('X2_data.npy').astype('float32')
X_feat = np.load('X2_features.npy').astype('float32')
y = np.load('Y2_labels.npy').astype('int')

# Combine signals and features to keep splitting consistent
X_combined = np.hstack([
    X_sig.reshape(len(X_sig), -1),
    X_feat
])

# First split: 70% training and 30% for validation + test
X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
    X_combined,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split: split the 30% equally into validation and test
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Apply SMOTE only on the training set
sm = SMOTE(random_state=42)
X_train_res, y_train = sm.fit_resample(X_train_raw, y_train_raw)

# Separate signals and features again, then reshape
def process_back(data):
    sigs = data[:, :120].reshape(-1, 1, 120)  # 1 channel
    feats = data[:, 120:]
    return sigs, feats

X_train_sig, X_train_feat = process_back(X_train_res)
X_val_sig, X_val_feat = process_back(X_val_raw)
X_test_sig, X_test_feat = process_back(X_test_raw)

# Smooth only the newly generated training samples
indices_new = np.arange(len(X_train_raw), len(X_train_res))

for i in indices_new:
    X_train_sig[i, 0, :] = np.convolve(
        X_train_sig[i, 0, :],
        np.ones(5) / 5,
        mode='same'
    )

# Combine signals with features into one matrix for the temporal model
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

