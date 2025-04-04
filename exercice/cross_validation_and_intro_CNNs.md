# Convolutional Neural Nets
A Convolutional Neural Network (CNN) is a specialized neural network designed for tasks involving structured grid data, particularly images. CNNs excel at capturing intricate features and patterns through their unique components:

1. **Convolutional Layers ($\text{Conv}$):** These layers employ a fundamental operation known as convolution to analyze input images. The convolutional operation involves sliding small learnable filters (also called kernels) across the input image, capturing local patterns at different spatial locations. Mathematically, the convolutional operation is expressed as:
   $$\text{Conv}(I, K) = I \ast K$$
   where $I$ is the input image, and $K$ is a learnable filter. The filters act as feature detectors, learning to recognize specific patterns, edges, or textures within the image (see `.gif` animation below for a visualization).

   The CNN learns these filters during the training process. Each filter specializes in detecting particular features, and through training, the network refines these filters to extract meaningful hierarchical representations from the input images.

2. **Activation Function ($\sigma(\cdot)$):** After each convolution operation, an activation function, typically ReLU  ($`\sigma(x) = \max(0, x)`$) , introduces non-linearity. This non-linearity is crucial for allowing the network to learn complex relationships and represent intricate features in the data.

3. **Pooling Layers ($\text{Pooling}$):** Following convolution, pooling layers reduce the spatial dimensions of the feature maps. Max pooling, for example, selects the maximum value from a group of neighboring pixels, aiding in retaining essential information while reducing computational load.

4. **Fully Connected Layers ($\text{FC}$):** The final layers of the CNN are fully connected, connecting neurons from the preceding layers to make decisions. These layers leverage the learned features to classify or regress on the input data.

In summary, the heart of a CNN lies in its convolutional layers, where learnable filters systematically analyze input images, capturing and refining intricate features that contribute to the network's ability to understand and interpret visual data.

![alt text](<Sans titre.gif>)


# Cross-Entropy Loss Explanation

## Introduction
Cross-entropy loss is a commonly used loss function for classification problems. It measures the dissimilarity between the predicted probability distribution and the true probability distribution of the labels. The goal is to minimize this loss so that the predicted probabilities align more closely with the actual class labels.

## Definition
For a given instance, let:
- $` y_i `$ be the true label (represented as a one-hot encoded vector).
- $` \hat{y}_i `$ be the predicted probability distribution over the classes.

The cross-entropy loss $` H(y, \hat{y}) `$ is defined as:


$$H(y, \hat{y}) = -\sum_{i} y_i \log(\hat{y}_i)$$


### Explanation of Terms
- $` y_i `$ is a one-hot encoded vector where only the index corresponding to the true class is 1, and all others are 0.
- $` \hat{y}_i `$ is the predicted probability for each class.
- The summation ensures that only the probability of the correct class contributes to the loss.

## Understanding the Loss Behavior
1. **Correct Predictions (Low Loss)**
   - If the model predicts a probability $` \hat{y}_i `$ close to 1 for the correct class, $` \log(\hat{y}_i) `$ is close to 0, leading to a small loss.

2. **Incorrect Predictions (High Loss)**
   - If the model assigns a very low probability to the correct class (i.e., is confidently wrong), $` \log(\hat{y}_i) `$ becomes a large negative number, resulting in a high loss.

### Example Calculation
Consider a classification task with three classes (A, B, C). If:
- The true class is **B** (one-hot vector: $` y = [0,1,0] `$).
- The model predicts probabilities **[0.7, 0.2, 0.1]**.

Then the loss is computed as:

$$H(y, \hat{y}) = - (0 \cdot \log(0.7) + 1 \cdot \log(0.2) + 0 \cdot \log(0.1)) = -\log(0.2)$$

Since $` \log(0.2) `$ is negative, the final loss value is positive and relatively large, indicating a poor prediction.

## Why Does High Confidence in a Wrong Prediction Result in High Loss?
If a model is **very confident but wrong**, the loss increases significantly. For example:
- If the model predicts **95%** probability for class A, but the true class is **C**, the loss is:
  
 $$ H(y, \hat{y}) = -\log(0.05) \approx 3$$
  
- This penalizes the model heavily, encouraging it to adjust its predictions to be more accurate.

## Conclusion
- Cross-entropy loss is used to evaluate how well the predicted probability distribution matches the true labels.
- It assigns **low loss** to correct and confident predictions and **high loss** to incorrect and confident predictions.
- The function helps train models by minimizing incorrect confident predictions and improving overall accuracy.

By minimizing this loss, the model learns to make more accurate predictions and avoid overconfident errors.

# K-fold Cross Validation

K-Fold Cross-Validation is a robust technique used to assess the performance and generalization of a machine learning model. It involves partitioning the dataset into k equally-sized folds and iteratively training and evaluating the model on different combinations of these folds. The key steps are as follows:

1. **Data Splitting:**
   - Divide the dataset into $`k`$ non-overlapping folds.
   - For each iteration, use $`k-1`$ folds for training and the remaining fold for testing.

2. **Model Training and Testing:**
   - Train the model on the $`k-1`$ training folds.
   - Evaluate the model on the held-out fold (testing set).

3. **Iteration:**
   - Repeat the process $`k`$ times, each time using a different fold as the testing set.
   - This ensures that each data point is used for testing exactly once.

4. **Performance Metrics:**
   - Record the performance metrics (e.g., accuracy, precision, recall) for each iteration.
   - Average the metrics across all iterations to obtain a more reliable estimate of the model's performance.

'Mathematically', the process can be represented as:
   $$\text{Fold}_{1} \rightarrow \text{Train}_{2,3,...,k} \quad \text{Test}_{1}$$
   $$\text{Fold}_{2} \rightarrow \text{Train}_{1,3,...,k} \quad \text{Test}_{2}$$
   $$\vdots$$
   $$\text{Fold}_{k} \rightarrow \text{Train}_{1,2,...,k-1} \quad \text{Test}_{k}$$

5. **Result Aggregation:**
   - Collect the performance metrics from each iteration.
   - Calculate the average and standard deviation of the metrics.

K-Fold Cross-Validation provides a more comprehensive evaluation of a model's robustness, reducing the impact of data variability and ensuring that the model's performance is consistent across different subsets of the data.