# Technical Report: Multimodal Network Analysis

## 1. Architecture Diagram
### CNN Classifier
For our CNN, we use a standard architecture that processes mel spectrograms as input. The architecture consists of three convolutional layers followed by fully connected layers for classification. Each convolutional layer is followed by batch normalization, ReLU activation, max pooling, and dropout for regularization.
This architecture was decided upon after some trial and error, as it provided a good balance between model complexity and performance on the validation set. The use of batch normalization helps stabilize training, while dropout prevents overfitting.

In preprocessing our audio, we first load it at a sampling rate of 22050Hz. We then pad each audio signal to match the length of the longest signal in the dataset. We then compute the mel spectrogram using the following parameters:
- **n_fft**: 1024 (the length of the FFT window)
- **hop_length**: 512 (the number of samples between successive frames)
- **n_mels**: 128 (the number of mel bands to generate)
This results in a mel spectrogram of shape (1, 128, 228) for each audio sample, which is the input to our CNN. The convolutional layers extract features from these spectrograms, while the fully connected layers perform the final classification into one of the 8 classes.
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
I first used OpenAI's Whisper model to transcribe the audio into text. The transcripts were then tokenized and fed into an RNN for classification.
Here I implemented a simple custom tokenizer to convert our transcripts into sequences of token IDs. The RNN architecture consists of an embedding layer followed by a two-layer LSTM and a final linear layer for classification. The embedding layer maps token IDs to dense vectors, while the LSTM captures temporal dependencies in the sequence data.
```
Input: Tokens (batch_size, seq_length, 768)
    |
Embedding Layer (vocab_size, embedding_dim=768)
    |
LSTM(input_size=768, hidden_size=128, num_layers=2, batch_first=True)
    |
Linear(128, 8)
```

### Late fusion of CNN and RNN
To combine the strengths of both the CNN and RNN classifiers, we implemented a late fusion strategy. In this approach, we take the output probabilities from both models and apply a weighted average to produce the final prediction. The weights were determined based on the validation performance of each model, giving more weight to the CNN due to its higher accuracy. This fusion allows us to leverage the complementary information captured by both models, resulting in improved overall performance on the test set.
```python
# Example of late fusion
cnn_probs = cnn_model.predict(X_cnn)  # Output probabilities from CNN
rnn_probs = rnn_model.predict(X_rnn)  # Output probabilities from RNN
combined_probs = 0.8 * cnn_probs + 0.2 * rnn_probs  # 80/20 weightage for late fusion
final_predictions = np.argmax(combined_probs, axis=1)  # Final class predictions
```

## 2. Results
### CNN Classifier
- Training Loss: 0.0937
- Validation Loss: ~1.5
- Test Accuracy: 55%

### RNN Classifier
- Training Loss: 1.9
- Validation Loss: ~2.0
- Test Accuracy: 22%

### Late Fusion
- Test Accuracy: 64%
Surprisingly higher than both the CNN and RNN, since the RNN is supposed to basically be a shot in the dark, but it seems to be providing some complementary information that the CNN is missing, which is why the late fusion is performing better than either model alone.

Plots are included in the ipython notebook for both training and validation losses, as well as test accuracies for each model.

# Mel Spectrograms and Waveforms
- A **waveform** is the raw audio signal plotted as amplitude over time. It shows how the sound pressure changes sample by sample and preserves the original time-domain structure of the audio.
- A **mel spectrogram** is a time-frequency representation computed from the waveform. It shows how audio energy is distributed across frequencies over time.
- The frequency axis uses the **mel scale**, which matches human hearing more closely by using finer resolution at lower frequencies and coarser resolution at higher frequencies.
- In this project, the CNN uses mel spectrograms because they convert audio into an image-like input that convolutional layers can process effectively.
- Compared with waveforms, mel spectrograms make recurring acoustic patterns easier to see and learn.
- These spectrograms are made using the **STFT** (Short-Time Fourier Transform), which is applying the **FFT** (Fast Fourier Transform) to short overlapping windows of the audio signal, allowing us to capture how frequency content changes over time.
