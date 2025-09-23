# Digit Recognition Neural Network

A handwritten digit recognition system built from scratch using a neural network with NumPy. This project implements a 3-layer neural network to classify handwritten digits (0-9) from the MNIST dataset.

## 🎯 Project Overview

This project demonstrates the implementation of a neural network for digit classification without using high-level machine learning frameworks like TensorFlow or PyTorch. The entire neural network is built using NumPy for educational purposes.

## 📁 Project Structure

```
Digit_Recoqnizer/
├── main.ipynb              # Main Jupyter notebook with implementation
├── test.py                 # Testing script
├── digit-recognizer/
│   ├── train.csv          # Training dataset (MNIST format)
│   └── test.csv           # Test dataset
└── README.md              # This file
```

## 🧠 Neural Network Architecture

- **Input Layer**: 784 neurons (28×28 pixel images flattened)
- **Hidden Layers**:
  - First Hidden Layer: 128 neurons with ReLU activation
  - Second Hidden Layer: 64 neurons with ReLU activation
- **Output Layer**: 10 neurons with Softmax activation (one for each digit 0-9)

### Key Components:

1. **Forward Propagation**: Computes predictions using current weights and biases
2. **Backward Propagation**: Calculates gradients for parameter updates
3. **Gradient Descent**: Updates weights and biases to minimize loss
4. **Activation Functions**:
   - ReLU for hidden layers
   - Softmax for output layer

## 🚀 Features

- **From-scratch implementation**: No high-level ML frameworks used
- **Data preprocessing**: Normalization and proper train/validation split
- **Visualization**: Display predictions with actual digit images
- **Training monitoring**: Real-time accuracy tracking during training

## 📊 Dataset

The project uses the MNIST handwritten digit dataset:
- **Training set**: 42,000 samples
- **Features**: 784 pixel values (28×28 images)
- **Labels**: Digits 0-9
- **Preprocessing**: Pixel values normalized to [0,1] range

## 🛠️ Implementation Details

### Data Preprocessing
```python
# Data is split into training (1000 samples) and validation sets
# Pixel values are normalized by dividing by 255
X_train = X_train / 255.0
```

### Network Initialization
```python
# Weights initialized randomly between -0.5 and 0.5
w1 = np.random.rand(128, 784) - 0.5
b1 = np.random.rand(128, 1) - 0.5
w2 = np.random.rand(64, 128) - 0.5
b2 = np.random.rand(64, 1) - 0.5
w3 = np.random.rand(10, 64) - 0.5
b3 = np.random.rand(10, 1) - 0.5
```

### Training Process
- **Learning rate**: 0.10
- **Iterations**: 1000
- **Batch processing**: Full batch gradient descent
- **Accuracy tracking**: Printed every 10 iterations

## 📈 Performance

The model achieves reasonable accuracy on the training set. Key improvements made:

1. **Expanded Network Architecture**: Added a second hidden layer for better feature extraction
2. **Fixed data preprocessing bugs**: Proper train/validation split and normalization
3. **Corrected bias gradients**: Fixed gradient calculation in backpropagation for better learning
4. **Improved softmax function**: Enhanced numerical stability for large values

## 🔧 Usage

1. **Setup Environment**:
   ```bash
   pip install numpy pandas matplotlib
   ```

2. **Run the Notebook**:
   - Open `main.ipynb` in Jupyter Notebook or VS Code
   - Run all cells sequentially

3. **Training the Model**:
   ```python
   w1, b1, w2, b2, w3, b3 = gradient_descent(X_train, Y_train, 0.10, 1000)
   ```

4. **Testing Predictions**:
   ```python
   test_prediction(index, w1, b1, w2, b2, w3, b3)
   ```

## 📝 Code Structure

### Cell 1: Library Imports
- NumPy for numerical operations
- Matplotlib for visualization
- Pandas for data handling

### Cell 2: Data Loading
- Loads the training dataset from CSV

### Cell 3: Data Preprocessing
- Converts to NumPy arrays
- Shuffles data randomly
- Splits into train/validation sets
- Normalizes pixel values

### Cell 4: Neural Network Functions
- Parameter initialization
- Activation functions (ReLU, Softmax)
- Forward propagation
- Backward propagation
- Parameter updates

### Cell 5: Training Functions
- Gradient descent implementation
- Accuracy calculation
- Prediction functions

### Cell 6: Model Training
- Trains the neural network
- Displays training progress

### Cells 7-8: Testing and Visualization
- Makes predictions on test samples
- Visualizes results with images

## 🔍 Key Functions

| Function | Purpose |
|----------|---------|
| `init_params()` | Initialize weights and biases randomly |
| `forward_prop()` | Compute forward pass through network |
| `back_prop()` | Calculate gradients via backpropagation |
| `gradient_descent()` | Train the network using gradient descent |
| `get_predictions()` | Get predicted class from probabilities |
| `test_prediction()` | Visualize prediction for a single sample |

## 🎓 Learning Objectives

This project demonstrates:
- Neural network fundamentals
- Gradient descent optimization
- Backpropagation algorithm
- Activation functions
- Data preprocessing techniques
- Model evaluation and visualization

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Optimize performance

## 📄 License

This project is for educational purposes. Feel free to use and modify as needed.

## 🙋‍♂️ Author

Created as part of a machine learning educational project to understand neural networks from first principles.

---

**Note**: This implementation prioritizes educational value and understanding over performance. For production use, consider using optimized frameworks like TensorFlow or PyTorch.
