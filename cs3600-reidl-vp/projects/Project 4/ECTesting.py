from NeuralNet import buildNeuralNet
import numpy as np

# Custom function to load XOR examples
def loadXORExamples():
    xor_examples = []
    with open('xor.txt', 'r') as file:
        for line in file:
            values = [int(x) for x in line.strip().split()]
            xor_examples.append(([values[0], values[1]], [values[2]]))
    return xor_examples

def testXOR(hiddenLayers):
    xor_examples = loadXORExamples()
    _, accuracy = buildNeuralNet((xor_examples, xor_examples), maxItr=1000, hiddenLayerList=hiddenLayers)
    return accuracy

# Test without a hidden layer
accuracy_no_hidden = testXOR([])

# Test with increasing numbers of perceptrons in the hidden layer
hidden_layer_sizes = list(range(11))  # Test up to 10 perceptrons
results = []

for num_hidden_layers in hidden_layer_sizes:
    accuracies = []
    for _ in range(5):  # Repeat the test 5 times for each configuration
        accuracy = testXOR([num_hidden_layers])
        accuracies.append(accuracy)
    max_accuracy = max(accuracies)
    avg_accuracy = np.mean(accuracies)
    std_deviation = np.std(accuracies)
    results.append((num_hidden_layers, max_accuracy, avg_accuracy, std_deviation))

# Print results
print(f"Accuracy without hidden layer: {accuracy_no_hidden:.4f}")
print("Perceptrons\tMax Accuracy\tAvg Accuracy\tStd Deviation")
for num_hidden_layers, max_accuracy, avg_accuracy, std_deviation in results:
    print(f"{num_hidden_layers}\t\t{max_accuracy:.4f}\t\t{avg_accuracy:.4f}\t\t{std_deviation:.4f}")
