import argparse
import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import csv

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--number", help="show specified number of lines", type=int)
parser.add_argument("-p", "--plot", help="enable or disable plots", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

SAMPLE_TEST = 'mnist_test50.csv'
SAMPLE_TEACH = 'mnist_test100.csv'

def show_train_data(filename):
    train_data = []
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            train_data.append(row)
    csv_file.close()

    length = len(train_data)
    rows = length // 10
    columns = length // rows

    fig, axs = plt.subplots(rows, columns, figsize=(18, 10), constrained_layout=True)
    num = 0
    for i in range(rows):
        for j in range(columns):
            image_array = np.asfarray(train_data[num][1:], dtype='int64').reshape((28, 28))
            axs[i][j].imshow(image_array, cmap='Greys', interpolation='None')
            num += 1

def test_network_with_numbers(lines_qtt, test_sample_file):
    test_data = []
    with open(test_sample_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            test_data.append(row)
    csv_file.close()

    if lines_qtt > len(test_data):
        lines_qtt = len(test_data)
    count = 0
    for line in test_data:
        count += 1

        scaled_input = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.01
        output = network.query(scaled_input)
        ind = np.argmax(output)

        print("Number: ", int(line[0]))
        print("It is: ", marks[ind])
        print("Output: ", output)

        line = np.asfarray(line[1:]).reshape((28, 28))
        plt.imshow(line, cmap='Greys', interpolation='None')
        plt.show()

        if count == lines_qtt:
            return

def test_network(test_sample_file):
    test_case = []

    test_data = []
    with open(test_sample_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            test_data.append(row)
    csv_file.close()
    
    for line in test_data:
        scaled_in = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.01
        out = network.query(scaled_in)
        i_max = np.argmax(out)

        if int(line[0]) == marks[i_max]:
            test_case.append(1)
        else:
            test_case.append(0)

    correct = sum(test_case)
    print("Correct values: ", correct)
    print("Incorrect values: ", len(test_case) - correct)
    print("Accuracy percentage: ", correct / len(test_case))

    return correct / len(test_case)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, teach_sample_filename):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.teach_sample_filename = teach_sample_filename

        self.wih = (np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5)
        self.who = (np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5)
        self.activation_function = lambda x: scipy.special.expit(x)
        self.counter = 0
        self.correct = 0

    def query(self, input_list):
        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def train(self, input_list, target_list, target):
        self.counter += 1

        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        i_max = np.argmax(final_outputs)
        if marks[i_max] == target:
            self.correct += 1

        return output_errors


input_nodes = 28 * 28
hidden_nodes = 100
output_nodes = 5
learning_rate = 0.2

total = 0
correct = 0

epoch_errors = []
epoch = 10

markers = {0: 0, 2: 1, 3: 2, 4: 3, 8: 4}
marks = [0, 2, 3, 4, 8]
target_values = []

network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, 0.2, SAMPLE_TEACH)

for mar in markers:
    target = np.zeros(output_nodes) + 0.01
    target[markers[mar]] = 0.99
    target_values.append(target)

test_data = []
with open(SAMPLE_TEST, 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        test_data.append(row)
csv_file.close()

delta = 0.000001
delta_count = 123123
previous = 10000000

for i in range(epoch):
    sum_err = 0
    for line in test_data:
        target = target_values[markers[int(line[0])]]

        scaled_input = (np.asfarray(line[1:]) / 255.0 * 0.99) + 0.01

        err = network.train(scaled_input, target, int(line[0]))
        err_norm = np.linalg.norm(err)
        sum_err += err_norm

    epoch_errors.append(sum_err / len(test_data))
    delta_count = previous - sum_err / len(test_data)
    previous = sum_err / len(test_data)

print("Training:")
print("Correct answers: ", network.correct)
print("Incorrect answers: ", network.counter - network.correct)
print("Accuracy percentage: ", network.correct / network.counter)

test_network(SAMPLE_TEST)

if args.number != None:
    test_network_with_numbers(args.number, SAMPLE_TEST)

if args.plot:
    fig1, ax1 = plt.subplots()
    ax1.plot(epoch_errors)
    stt = 'Epoch Standard Deviation'
    ax1.set_title(stt)
    ax1.grid(True)

    show_train_data(SAMPLE_TEACH)
    show_train_data(SAMPLE_TEST)

    plt.show()
