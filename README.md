# 🧠 Feedforward Neural Network in C99

Welcome to the **Feedforward Neural Network** project! This project is a simple yet powerful implementation of a **neural network** built in **C99** to classify **28x28 pixel images of digits**. 📊

PS. Everything I do is on WSL2 runnig Debian 12 (Bookworm)

---

## 🚀 Features

- 🖼️ **Image Input**: 28x28 grayscale images (e.g., from the MNIST dataset).
- ⚡ **Activation Function**: ReLU (Rectified Linear Unit) for non-linear activation.
- 🔄 **Learning Method**: Backpropagation for optimizing the network.
- 🧮 **Loss Function**: Cross-entropy to calculate the model's performance.
- 🔧 **Adjustable Hyperparameters**: Easily customize layer sizes, learning rates, etc.

---

## 📂 Project Structure

```bash
.
├── src
│   ├── main.c       
│   └── softmax.c
├── include
│   └── softmax.h
├── data
│   ├── data_organizer.py
│   ├── test_images.txt
│   ├── test_lables.txt
│   ├── training_images.txt
│   └── training_lables.txt
├── .gitignore
├── makefile
├── LICENSE
└── README.md        # You're looking at it! 📄
```

## 🛠️ Installation
1. Clone the Repository:
```bash
git clone https://github.com/username/feedforward-neural-network.git
cd feedforward-neural-network
```
2. Run the makefile:
```bash
make
```
OR

Compile the Code:
Make sure you have a C99-compatible compiler (like gcc):
```bash
gcc -std=c99 -o prg src/main.c src/softmax.c -Iinclude -lm
```
3. Run the Neural Network:
```bash
./prg
```
---

## 🧑‍🏫 How It Works

1. **Input Layer**:  
   The network takes **28x28 pixel images** (flattened to a **784-dimensional vector**) as input. Each pixel is normalized to the range [0, 1].

2. **Hidden Layers**:  
   Uses **fully connected hidden layers** with **ReLU activation**. The number of hidden layers and neurons can be easily adjusted in the configuration.

3. **Output Layer**:  
   The output layer contains **10 neurons** corresponding to the 10 digit classes (0-9). The **softmax function** is applied to convert raw scores into probabilities.

4. **Training**:  
   The network uses **backpropagation** and the **cross-entropy loss function** to update weights and biases during training.

---

## 📊 Performance

🧑‍💻 After training for several epochs, the neural network can achieve **high accuracy** on the test set of digit images. You can tweak the learning rate, number of layers, or number of neurons to improve performance.

---

## 🛠️ Hyperparameters

- **Learning Rate**: Adjustable in the `main.c`.
- **Number of Hidden Layers**: Configurable in the `main.c`.
- **Number of Neurons**: You can set the number of neurons for each hidden layer.

---

## 📈 Results

After training, you should expect to see results like:

- **Training Accuracy**: ~98%
- **Test Accuracy**: ~97%

---

## 🧑‍💻 Contributing

We welcome contributions from the community! Feel free to submit **pull requests** or open **issues**.

---

## 📝 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

Happy coding! 😎
















