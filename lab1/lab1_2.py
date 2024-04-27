import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--plot", help="enable or disable plots", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

CHR_LEN = 70
BIT_LEN = 33

K_CS_COUNT = 3
PSL = 3

P_C = 0.9
P_M = 0.2

def ACF(k, a):
    psls = []

    for k in range(-CHR_LEN, CHR_LEN):
        if k < -BIT_LEN-1 or k > BIT_LEN-1:
            psls.append(0.0)
            continue
        if k <= 0:
            cur = np.sum([a[i]*a[i-k] for i in range(0, BIT_LEN+k)])
        else:
            cur = np.sum([a[i]*a[i-k] for i in range(BIT_LEN-1, -1+k, -1)])

        psls.append(cur)
    
    return psls

def FFn(a):
    return BIT_LEN / PSL(a)

def PSL(a):
    cur_max = -9999
    psls = []

    for k in range(-CHR_LEN, CHR_LEN):
        if k < -BIT_LEN-1 or k > BIT_LEN-1:
            psls.append(0.0)
            continue
        if k == 0:
            continue
        if k < 0:
            cur = np.sum([a[i]*a[i-k] for i in range(0, BIT_LEN+k)])
        else:
            cur = np.sum([a[i]*a[i-k] for i in range(BIT_LEN-1, -1+k, -1)])

        psls.append(cur)
        if cur >= cur_max:
            cur_max = cur

    return cur_max

def generate_sample(n, p):
    sample = []
    rng = np.random.default_rng()

    for i in range(p):
        chromosomes = rng.integers(2, size=n)
        chromosomes = list(map(lambda chr: -1 if chr==0 else 1, chromosomes))
        sample.append(chromosomes)
    
    return sample

def select_pairs(sample):
    pairs = []
    s = sample
    for _ in range(s.size):
        selected = np.random.choice(s, size=2, replace=False)
        pairs.append(selected)
        s = np.setdiff1d(sample, selected)
    return np.array(pairs)

def crossover(pair):
    k = np.random.randint(0, BIT_LEN)

    x1i = pair[0][0:k]
    x1j = pair[0][k:]
    x2i = pair[1][0:k]
    x2j = pair[1][k:]

    new_first = mutate(np.concatenate((x1i, x2j)))
    new_second = mutate(np.concatenate((x2i, x1j)))

    return np.array([new_first, new_second])

def mutate(child):
    m = np.random.rand()
    if m < P_M:
        n = np.random.randint(0, BIT_LEN)
        if child[n] == 1:
            child[n] = -1
        else:
            child[n] = 1 
    return child

def reduce_population(sample):
    new_sample = np.array([])

    for i in range(CHR_LEN):
        Fs = 0
        for a in sample:
            Fs += FFn(a)
        
        sectors = []
        Fo_prev = 0
        for a in sample:
            Fo = FFn(a) / Fs
            sector = Fo + Fo_prev
            Fo_prev = sector
            sectors.append(sector)
        sectors.append(1)

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
        pb1 = pair[0]
        pb2 = pair[1]

        s = np.random.rand()
        if s < P_C:
            children = crossover([pb1, pb2])
            sample = np.append(sample, children[0])
            sample = np.append(sample, children[1])

    new_generation = reduce_population(sample=sample)

    return new_generation

def genetic_algorithm(sample):
    generation_count = 0
    generation = np.array([])

    Fs = 0
    Fs_prev = 0
    dF = 100

    for num in sample:
        Fs += FFn(num)
    print("----------\nGeneration #%d\nFs current: %.4f\nFs previous: %.4f\ndF: %.4f\n" % (generation_count, Fs, Fs_prev, dF))

    # while generation_count != 100:
    while dF > 0.1:
        generation_count += 1
        generation = new_generation(sample=sample)

        Fs = 0
        for num in generation:
            Fs += FFn(num)

        dF = (Fs - Fs_prev)/ Fs
                    
        print("----------\nGeneration #%d\nFs current: %.4f\nFs previous: %.4f\ndF: %.4f\n" % (generation_count, Fs, Fs_prev, dF))

        Fs_prev = Fs

    m = np.max([FFn(num) for num in generation])
    print(m)

    return generation

def print_sample_table(sample):
    print("+ {sign:.2s} + {sign:132s} + {sign:.7s} +".format(sign=("-" * 132)))
    print("| {:^2s} | {:^132s} | {:^7s} |".format("№", "Chromosome", "PSL"))
    print("+ {sign:.2s} + {sign:.132s} + {sign:.7s} +".format(sign=("-" * 132)))
    
    i = 0
    for a in sample:
        print("| {:^2d} | {:<132s} | {:^7d} |".format(i, ', '.join(str(x) for x in a), PSL(a)))
        i += 1
    print("+ {sign:.2s} + {sign:.132s} + {sign:.7s} +".format(sign=("-" * 132)))

first_sample = generate_sample(BIT_LEN, CHR_LEN)
print_sample_table(first_sample)

if args.plot:
    
    

    plt.show()
