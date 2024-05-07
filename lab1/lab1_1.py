import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--plot", help="enable or disable plots", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

CHR_LEN = 50
BIT_LEN = 20

LOW = -3
HIGH = 3

P_C = 0.85
P_M = 0.15

def f(x):
    return np.cos(1.2 * x - 2) - np.cos(1.7 * x - 1) * np.sin(8.4 * x)

def F(x):
    return np.cos(1.2 * x - 2) - np.cos(1.7 * x - 1) * np.sin(8.4 * x) + 2

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
    k = np.random.randint(0, BIT_LEN)

    x1i = pair_in_binary[0][0:k]
    x1j = pair_in_binary[0][k:]
    x2i = pair_in_binary[1][0:k]
    x2j = pair_in_binary[1][k:]

    new_first = mutate(x1i + x2j)
    new_second = mutate(x2i + x1j)

    return np.array([new_first, new_second])


def mutate(child_in_binary):
    m = np.random.rand()
    str_list = list(child_in_binary)
    if m < P_M:
        n = np.random.randint(0, BIT_LEN)
        if str_list[n] == "1":
            str_list[n] = "0"
        else:
            str_list[n] = "1" 
    return "".join(str_list)

def reduce_population(sample):
    new_sample = np.array([])

    for i in range(CHR_LEN):
        Fs = 0
        for num in sample:
            Fs += F(num)
        
        sectors = []
        Fo_prev = 0
        for num in sample:
            Fo = F(num) / Fs
            sector = Fo + Fo_prev
            Fo_prev = sector
            sectors.append(sector)

        c = np.random.rand()
        for i, sector in enumerate(sectors):
            if c < sector:
                new_sample = np.append(new_sample, sample[i])
                sample = np.delete(sample, i)
                break

    return new_sample
    
def new_generation(sample):
    pairs = select_pairs(sample=sample)

    for pair in pairs:
        pb1 = float_to_binary(pair[0], BIT_LEN, LOW, HIGH)
        pb2 = float_to_binary(pair[1], BIT_LEN, LOW, HIGH)

        s = np.random.rand()
        if s < P_C:
            children = crossover([pb1, pb2])
            sample = np.append(sample, binary_to_float(children[0], BIT_LEN, LOW, HIGH))
            sample = np.append(sample, binary_to_float(children[1], BIT_LEN, LOW, HIGH))

    new_generation = reduce_population(sample=sample)

    return new_generation

def genetic_algorithm(sample):
    generation_count = 0
    generation = np.array([])
    generation_previous = sample

    Fs = 0
    Fs_prev = 0
    dF = 100

    for num in sample:
        Fs += F(num)
    print("----------\nGeneration #%d\nFs current: %.4f\nFs previous: %.4f\ndF: %.4f\n" % (generation_count, Fs, Fs_prev, dF))

    # while generation_count != 100:
    while dF > 0.0001:
        generation_count += 1
        generation = new_generation(sample=generation_previous)

        Fs = 0
        for num in generation:
            Fs += F(num)

        dF = (Fs - Fs_prev)/ Fs
                    
        print("----------\nGeneration #%d\nFs current: %.4f\nFs previous: %.4f\ndF: %.4f\n" % (generation_count, Fs, Fs_prev, dF))

        Fs_prev = Fs
        generation_previous = generation

    m = np.max([F(num) for num in generation])
    print(m)

    return generation

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

    plt.figure(figure)
    plt.title("Функция приспособленности")
    plt.xlabel("x")
    plt.ylabel("F(x)")
    if sample is None:
        plt.plot(x, F(x))
    else:
        plt.plot(x, F(x), sample, F(sample), "co")

def plot_target_function(figure):
    step = 0.01
    x = np.arange(LOW, HIGH, step)

    plt.figure(figure)
    plt.title("Целевая функция")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, f(x))


rng = np.random.default_rng()

first_sample = rng.uniform(LOW, HIGH, CHR_LEN)
second_sample = rng.uniform(LOW, HIGH, CHR_LEN)
third_sample = rng.uniform(LOW, HIGH, CHR_LEN)

print_sample_table(first_sample)
final_generation1 = genetic_algorithm(sample=first_sample)

print_sample_table(second_sample)
final_generation2 = genetic_algorithm(sample=second_sample)

print_sample_table(third_sample)
final_generation3 = genetic_algorithm(sample=third_sample)

if args.plot:
    
    plot_target_function("Target function")

    # plot_fitness_function(1, sample=None)

    plot_fitness_function("1st sample before", sample=first_sample)
    plot_fitness_function("1st sample after", sample=final_generation1)

    plot_fitness_function("2d sample before", sample=second_sample)
    plot_fitness_function("2d sample after", sample=final_generation2)

    plot_fitness_function("3d sample before", sample=third_sample)
    plot_fitness_function("3d sample after", sample=final_generation3)

    plt.show()
