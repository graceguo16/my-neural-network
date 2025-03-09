
Neural Network Implementation from Scratch 🧠
Hi! This is my first time construct neural network functions in python. It contains an implementation of a simple neural network built from scratch using NumPy. The network consists of an input layer, a hidden layer with two neurons, and an output layer with a single neuron.

📂 Project Structure
neural_network.py - Defines the OurNeuralNetwork class, implementing a basic feedforward neural network with backpropagation.
neuron.py - Implements a single neuron class, demonstrating the core computations of a neural network.
loss_functions.py - Contains the Mean Squared Error (MSE) loss function used to evaluate the network’s performance.
train1.py - Script for training the neural network on a small dataset and monitoring the loss.
🚀 Getting Started
1️⃣ Install Dependencies
Ensure you have Python installed, then install NumPy and Matplotlib:
pip install numpy matplotlib
2️⃣ Run the Training Script
To train the neural network, simply execute:
python train1.py
This script will:
Train the network using a small dataset (predicting gender based on weight & height).
Output the loss at every 10 epochs.

📊 Visualizing Training Loss (Optional)
If you want to track how the loss decreases over time, modify the training script (train1.py) to store the loss at each epoch and plot it using matplotlib. Example:
import matplotlib.pyplot as plt

losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for x, y_true in zip(data, all_y_trues):
        y_pred = network.feedforward(x)
        loss = mse_loss(np.array([y_true]), np.array([y_pred]))
        epoch_loss += loss
    losses.append(epoch_loss / len(data))

plt.plot(range(epochs), losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.savefig("training_loss.png")
plt.show()

📝 Future Improvements
Implement batch training for improved performance.
Explore different activation functions (ReLU, Tanh).
Extend to multi-class classification.


