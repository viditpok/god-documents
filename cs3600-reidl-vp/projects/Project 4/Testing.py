from NeuralNetUtil import buildExamplesFromCarData, buildExamplesFromPenData
from NeuralNet import buildNeuralNet
from math import pow, sqrt

def average(argList):
    return sum(argList) / float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val - mean), 2) for val in argList]
    return sqrt(sum(diffSq) / len(argList))

def run_tests(data_func, num_hidden_layers):
    accuracies = []
    for _ in range(5):
        _, accuracy = data_func(hiddenLayers=num_hidden_layers)
        accuracies.append(accuracy)
    return accuracies

def testPenData(hiddenLayers=[24]):
    penData = buildExamplesFromPenData()
    return buildNeuralNet(penData, maxItr=200, hiddenLayerList=hiddenLayers)

def testCarData(hiddenLayers=[16]):
    carData = buildExamplesFromCarData()
    return buildNeuralNet(carData, maxItr=200, hiddenLayerList=hiddenLayers)

# Vary the number of perceptrons in the hidden layer from 0 to 40 in increments of 5
hidden_layers_list = list(range(0, 45, 5))

# Store max, average, and standard deviation for each number of perceptrons
results = []

for num_hidden_layers in hidden_layers_list:
    pen_accuracies = run_tests(testPenData, [num_hidden_layers])
    car_accuracies = run_tests(testCarData, [num_hidden_layers])

    max_pen_accuracy = max(pen_accuracies)
    avg_pen_accuracy = average(pen_accuracies)
    stdev_pen_accuracy = stDeviation(pen_accuracies)

    max_car_accuracy = max(car_accuracies)
    avg_car_accuracy = average(car_accuracies)
    stdev_car_accuracy = stDeviation(car_accuracies)

    results.append({
        "Hidden Layers": num_hidden_layers,
        "Max Pen Accuracy": max_pen_accuracy,
        "Avg Pen Accuracy": avg_pen_accuracy,
        "StDev Pen Accuracy": stdev_pen_accuracy,
        "Max Car Accuracy": max_car_accuracy,
        "Avg Car Accuracy": avg_car_accuracy,
        "StDev Car Accuracy": stdev_car_accuracy
    })

# Print the results in a table format
print("Hidden Layers | Max Pen Accuracy | Avg Pen Accuracy | StDev Pen Accuracy | Max Car Accuracy | Avg Car Accuracy | StDev Car Accuracy")
for result in results:
    print(f"{result['Hidden Layers']:13} | {result['Max Pen Accuracy']:15.5f} | {result['Avg Pen Accuracy']:15.5f} | {result['StDev Pen Accuracy']:18.5f} | {result['Max Car Accuracy']:15.5f} | {result['Avg Car Accuracy']:15.5f} | {result['StDev Car Accuracy']:18.5f}")
