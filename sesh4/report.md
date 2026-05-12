# Technical Report: Multimodal Network Analysis

## 1. Architecture Diagram

### CNN Classifier
```
Input: Mel Spectrogram (1, 128, 228)
    |
Conv2D(1, 32, kernel_size=3, stride=1, padding=1) -> BatchNorm2d(32) -> ReLU() -> MaxPool2d(kernel_size=2, stride=2) -> Dropout(0.2)
    |
Conv2D(32, 64, kernel_size=3, stride=1, padding=1) -> BatchNorm2d(64) -> ReLU() -> MaxPool2d(kernel_size=2, stride=2) -> Dropout(0.2)
    |
Conv2D(64, 64, kernel_size=3, stride=1, padding=1) -> BatchNorm2d(64) -> ReLU() -> MaxPool2d(kernel_size=2, stride=2) -> Dropout(0.2)
    |
Flatten()
    |
Linear(28672, 512) -> ReLU() -> Linear(512, 128) -> ReLU() -> Dropout(0.4) -> Linear(128, 8)
```

### RNN Classifier
```
Input: Tokens (batch_size, seq_length, 768)
    |
Embedding Layer (vocab_size, embedding_dim=768)
    |
LSTM(input_size=768, hidden_size=128, num_layers=2, batch_first=True)
    |
Linear(128, 8)
```

![wefwef](fwefwef)
## 2. Results
### CNN Classifier
- Training Loss: 0.0937
- Validation Loss: ~1.5
- Test Accuracy: 55%
