import numpy as np
import matplotlib.pyplot as plt

PLOT = False

CHR_LEN = 50
BIT_LEN = 20

LOW = -3
HIGH = 3

P_C = 0.85
P_M = 0.15

def f(x):
    return np.cos(1.2 * x - 2) - np.cos(1.7 * x - 1) * np.sin(8.4 * x)

def F(x):
    return np.cos(1.2 * x - 2) - np.cos(1.7 * x - 1) * np.sin(8.4 * x)

def compare_floats(a, b, tolerance=1e-9):
    return abs(a - b) < tolerance

def float_to_binary(number, length, min_val, max_val):
    number = max(min(number, max_val), min_val)

    normalized_number = (number - min_val) / (max_val - min_val)

    binary_str = format(int(normalized_number * (2**length)), f'0{length}b')

    return binary_str

def binary_to_float(binary_str, length, min_val, max_val):
    int_value = int(binary_str, 2)

    normalized_value = int_value / (2**length)

    value = min_val + normalized_value * (max_val - min_val)

    return value

def select_pairs(sample):
    pairs = []
    s = sample
    for _ in range(s.size):
        selected = np.random.choice(s, size=2, replace=False)
        pairs.append(selected)
        s = np.setdiff1d(sample, selected)
    return np.array(pairs)

def crossover(pair_in_binary):
    s = np.random.rand()
    if s < P_C:
        k = np.random.randint(0, BIT_LEN)

        x1i = pair_in_binary[0][0:k]
        x1j = pair_in_binary[0][k+1:]
        x2i = pair_in_binary[1][0:k]
        x2j = pair_in_binary[1][k+1:]

        new_first = mutate(x1i + x2j)
        new_second = mutate(x2i + x1j)

        return [new_first, new_second]
    else:
        return np.array(pair_in_binary)

def mutate(child_in_binary):
    m = np.random.rand()
    if m < P_M:
        n = np.random.randint(0, BIT_LEN)
        if child_in_binary[n] == 1:
            child_in_binary[n] = 0
        else:
            child_in_binary[n] = 1 
    return child_in_binary

def genetic_algorithm(sample):
    pairs = select_pairs(sample=sample)

    for pair in pairs:
        pb1 = float_to_binary(pair[0], BIT_LEN, LOW, HIGH)
        pb2 = float_to_binary(pair[1], BIT_LEN, LOW, HIGH)

        children = crossover([pb1, pb2])
        if children != pair:
            sample.append(binary_to_float(children[0], BIT_LEN, LOW, HIGH))
            sample.append(binary_to_float(children[1], BIT_LEN, LOW, HIGH))



def print_sample_table(sample):
    print("+ {sign:.2s} + {sign:.7s} + {sign:.20s} + {sign:.7s} + {sign:.7s} +".format(sign=("-" * 20)))
    print("| %2s | %7s | %20s | %7s | %7s |" % ("№", "X", "Chromosome", "f(x)", "F(x)"))
    print("+ {sign:.2s} + {sign:.7s} + {sign:.20s} + {sign:.7s} + {sign:.7s} +".format(sign=("-" * 20)))
    for i, num in enumerate(sample):
        print("| %2d | %+1.4f | %20s | %+1.4f | %+1.4f |" % 
              (i, num, float_to_binary(num, 20, LOW, HIGH), f(num), F(num)))
    print("+ {sign:.2s} + {sign:.7s} + {sign:.20s} + {sign:.7s} + {sign:.7s} +".format(sign=("-" * 20)))

def plot_fitness_function(figure, sample):
    step = 0.01
    x = np.arange(LOW, HIGH, step)

    plt.title("Функция приспособленности")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.figure(figure)
    if sample is None:
        plt.plot(x, F(x))
    else:
        plt.plot(x, F(x), sample, F(sample), "co")

def plot_target_function(figure):
    step = 0.01
    x = np.arange(LOW, HIGH, step)

    plt.title("Целевая функция")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.figure(figure)
    plt.plot(x, f(x))


rng = np.random.default_rng()

first_sample = rng.uniform(LOW, HIGH, CHR_LEN)
second_sample = rng.uniform(LOW, HIGH, CHR_LEN)
third_sample = rng.uniform(LOW, HIGH, CHR_LEN)

print_sample_table(first_sample)

if PLOT:
    plot_target_function(0)
    plot_fitness_function(1, sample=None)
    plot_fitness_function(2, sample=first_sample)
    plot_fitness_function(3, sample=second_sample)
    plot_fitness_function(4, sample=third_sample)

    plt.show()
