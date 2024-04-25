import numpy as np
import matplotlib.pyplot as plt


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

def float_to_binary_fixed_point(number, length, min_val, max_val):

    number = max(min(number, max_val), min_val)

    normalized_number = (number - min_val) / (max_val - min_val)

    binary_str = format(int(normalized_number * (2**length)), f'0{length}b')

    return binary_str

def print_sample_table(sample):
    print("+ {sign:.2s} + {sign:.7s} + {sign:.20s} + {sign:.7s} + {sign:.7s} +".format(sign=("-" * 20)))
    print("| %2s | %7s | %20s | %7s | %7s |" % ("№", "X", "Chromosome", "f(x)", "F(x)"))
    print("+ {sign:.2s} + {sign:.7s} + {sign:.20s} + {sign:.7s} + {sign:.7s} +".format(sign=("-" * 20)))
    for i, num in enumerate(sample):
        print("| %2d | %+1.4f | %20s | %+1.4f | %+1.4f |" % 
              (i, num, float_to_binary_fixed_point(num, 20, LOW, HIGH), f(num), F(num)))
    print("+ {sign:.2s} + {sign:.7s} + {sign:.20s} + {sign:.7s} + {sign:.7s} +".format(sign=("-" * 20)))

rng = np.random.default_rng()

first_sample = rng.uniform(LOW, HIGH, CHR_LEN)

print_sample_table(first_sample)




step = 0.01
x = np.arange(LOW, HIGH, step)

plt.title("Целевая функция")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.figure(0)
plt.plot(x, f(x))

plt.title("Функция приспособленности")
plt.xlabel("x")
plt.ylabel("F(x)")
plt.figure(1)
plt.plot(x, F(x))

plt.show()
