# Report: Recommender System Models and Approaches

## Overview

This task explored three recommender system approaches for implicit-feedback interaction data:

1. **Matrix Factorization (MF)**
2. **Multi-Layer Perceptron (MLP)-based recommender**
3. **Neural Collaborative Filtering (NCF)**, which combines MF and MLP ideas

The goal was to predict whether a user would interact with an item using a binary classification setup, where observed interactions were treated as positive examples and unobserved pairs were sampled as negatives.

## Data Preparation

The data consisted of user-item interaction pairs stored in `interactions.csv`. Since the dataset only contained positive interactions, negative sampling was required to build a training set suitable for supervised learning.

### Negative Sampling

For each observed `(user, item)` pair, the approach:
- kept the original interaction as a **positive sample** with label `1`
- sampled `num_neg` items the same user had not interacted with
- added those sampled pairs as **negative samples** with label `0`

This creates a balanced training signal for learning implicit feedback models. A lookup set of user-item interactions was used to avoid sampling items the user had already seen.

A good representation of our dataset should have a higher amount of negative samples than positive ones, as users typically interact with only a small fraction of available items. I experimented with different values of `num_neg` and found that `20` provided a good balance between training signal and computational efficiency.

## Model 1: Matrix Factorization (MF)

Matrix Factorization is the simplest baseline in this task. It represents users and items as learned embeddings and predicts preference using their dot product.

### Method

- Each user is mapped to a trainable embedding vector
- Each item is mapped to a trainable embedding vector
- The interaction score is computed as the dot product between the two vectors

### Strengths and Limitations

**Strengths:**
- Simple and fast to train
- Easy to interpret
- Often strong for sparse collaborative filtering problems

**Limitations:**
- Only models linear interactions
- Cannot easily capture more complex patterns in user preference

## Model 2: MLP-Based Recommender

The MLP model extends MF by learning a nonlinear interaction function. Instead of using only a dot product, it concatenates the user and item embeddings and passes them through several fully connected layers.

### Method

- Learn user and item embeddings
- Concatenate the embeddings
- Feed the combined vector into a multi-layer perceptron
- Output a single interaction score

### Strengths and Limitations

**Strengths:**
- More expressive than MF
- Can capture nonlinear patterns
- Flexible architecture with tunable depth and width

**Limitations:**
- More parameters than MF
- Higher risk of overfitting
- Usually requires more training and tuning

## Model 3: Neural Collaborative Filtering (NCF)

NCF combines the linear modeling power of MF with the nonlinear modeling power of the MLP. The implementation in this task computes both MF-style and MLP-style signals, then combines them to produce the final prediction.

### Strengths and Limitations

**Strengths:**
- More expressive than either MF or MLP alone
- Can leverage complementary signals

**Limitations:**
- More complex architecture
- Harder to tune and train
- Benefits may depend on dataset size and sparsity

## Training Procedure

All models were trained using the same general pipeline:

- PyTorch `Dataset` and `DataLoader` for batching
- `BCEWithLogitsLoss` as the loss function
- Adam optimizer with learning rate `0.001`
- Epoch-based optimization with backpropagation

The models output raw scores, which are suitable for `BCEWithLogitsLoss` because the loss internally applies a sigmoid-style transformation.

## Comparison

### MF vs MLP

MF is a strong baseline and works well when latent similarity is sufficient. The MLP is more expressive and can outperform MF when the interaction structure is more complex. However, MLP may also overfit more easily, especially with limited data.

### NCF

NCF is designed to improve over both baselines by combining linear and nonlinear signals. In practice, its performance depends on how well the architecture is balanced and how effectively it is trained.

### Effect of Embedding Size

Embedding dimension controls model capacity. Larger embeddings can capture more information but may also increase overfitting and computation cost. Smaller embeddings are cheaper but may underfit. I tried various embedding dimensions and decided to go with `128` as a good balance for this dataset.
```
