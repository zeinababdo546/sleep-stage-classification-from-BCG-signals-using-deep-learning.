# Dataset that loads 5 consecutive windows
class SleepSequenceDataset(Dataset):
    def __init__(self, X, y, seq_len=5):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]

train_loader = DataLoader(
    SleepSequenceDataset(X_train_final, y_train),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    SleepSequenceDataset(X_val_final, y_val),
    batch_size=64,
    shuffle=False
)

# Model (TemporalSleepModel)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        return F.relu(out)

class TemporalSleepModel(nn.Module):
    def __init__(self, num_classes=5):
        super(TemporalSleepModel, self).__init__()

        self.conv1 = nn.Conv1d(
            1,
            16,
            kernel_size=7,
            stride=1,
            padding=3
        )

        self.bn1 = nn.BatchNorm1d(16)

        self.layer1 = ResidualBlock(16, 16)
        self.layer2 = ResidualBlock(16, 32, stride=2)
        self.layer3 = ResidualBlock(32, 64, stride=2)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        b, s, c, l = x.size()

        x = x.view(b * s, c, l)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x).view(b, s, -1)

        x, _ = self.lstm(x)

        return self.fc(x[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TemporalSleepModel().to(device)
