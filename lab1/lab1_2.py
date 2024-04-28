import argparse
import matplotlib.pyplot as plt
import random
import copy

parser = argparse.ArgumentParser()

parser.add_argument("-p", "--plot", help="enable or disable plots", action=argparse.BooleanOptionalAction)

args = parser.parse_args()

CHR_LEN = 70
BIT_LEN = 33

K_CS_COUNT = 3
PSL = 3

P_C = 0.9
P_M = 0.2

ITERATIONS = 35

def ACF(chromosome):
    psls = []
    for k in range(-(BIT_LEN-1), BIT_LEN):
        count = 0
        i1 = k
        i2 = 0
        while i1 < BIT_LEN:
            if i2 >= BIT_LEN:
                break
            while i1 < 0:
                i2 += 1
                i1 += 1
            if chromosome[i2] == chromosome[i1]:
                count += 1
            else:
                count -= 1
            i2 += 1
            i1 += 1
        psls.append(count)
    
    return psls


def ACF_c(chromosomes, chromosome):
    for chrom in chromosomes:
        if chrom == chromosome:
            return False
        
    chromosome_list = list(chromosome)
    for i in range(0, BIT_LEN):
        chromosome_list[i] = str(int(chromosome_list[i]) ^ 1)

    chr_comparison = ''.join(chromosome_list)

    for chrom in chromosomes:
        if chr_comparison == chrom:
            return False
        
    return True


class CodeSequence:
    num = 1
    chromosome = ""
    psl_max = -9999

    def __init__(self, number):
        self.num = number

        bits = bin(number)
        bits = bits[2:]

        while len(bits) < BIT_LEN:
            bits = '0' + bits

        self.chromosome = bits

    def chr_from_string(self, chromosome):
        self.chromosome = chromosome

        n = 0
        length = BIT_LEN-1

        for ch in chromosome:
            n += int(ch)*(2**length)
            length = length-1

        self.num = n

    def FFn(self):
        cur_max = -9999
        
        if self.psl_max != cur_max:
            return BIT_LEN/self.psl_max
        
        for i in ACF(self.chromosome):
            if cur_max < i and i != BIT_LEN:
                cur_max = i
        self.psl_max = cur_max

        return BIT_LEN/cur_max

    def PSL(self):
        cur_max = -9999

        if self.psl_max != cur_max:
            return self.psl_max
        
        for i in ACF(self.chromosome):
            if cur_max < i and i != BIT_LEN:
                cur_max = i
        self.psl_max = cur_max

        return cur_max

    def get_chr(self):
        chromosome_list = list(self.chromosome)

        for i in range(0, BIT_LEN):
            if (chromosome_list[i] == '1'):
                chromosome_list[i] = '+1'
            else:
                chromosome_list[i] = '-1'
                
        return ', '.join(chromosome_list)


class Population:
    sample = []
    N = 1

    def __init__(self, n):
        self.N = n

        for _ in range(1, self.N+1):
            self.sample.append(CodeSequence(random.randint(0, 2 ** BIT_LEN)))

    def mutate(self, chromosome: str):
        m = random.randint(1, BIT_LEN)

        chromosome_list = list(chromosome)
        chromosome_list[m - 1] = str(int(chromosome_list[m - 1]) ^ 1)
        chromosome = ''.join(chromosome_list)

        return chromosome

    def crossover(self):
        pairs = copy.deepcopy(self.sample)

        while len(pairs) > 1:
            par1 = pairs.pop(random.randint(0, len(pairs)-1))
            par2 = pairs.pop(random.randint(0, len(pairs)-1))

            s = random.random()
            if s > P_C:
                continue

            k_crossover = random.randint(0, BIT_LEN)

            child1 = par1.chromosome[0:k_crossover] + par2.chromosome[k_crossover:]
            child2 = par2.chromosome[0:k_crossover] + par1.chromosome[k_crossover:]

            m = random.random()
            if m < P_M:
                child1 = self.mutate(child1)

            m = random.random()
            if m < P_M:
                child2 = self.mutate(child2)

            new_first = CodeSequence(1)
            new_first.chr_from_string(child1)
            new_second = CodeSequence(1)
            new_second.chr_from_string(child2)

            self.sample.append(new_first)
            self.sample.append(new_second)

    def reduction(self):
        pairs = copy.deepcopy(self.sample)
        count = 0
        new_sample = []

        while count < self.N:
            Fs = self.Fs()
            sectors = []

            for pair in pairs:
                sectors.append(pair.FFn()/Fs)

            c = random.random()
            i = 0
            if c < sectors[0]:
                new_sample.append(pairs.pop(0))
            else:
                while c > 0:
                    c -= sectors[i]
                    i += 1
                new_sample.append(pairs.pop(i))
            count = count+1

        self.sample = new_sample

    def Fs(self):
        count = 0

        for i in range(0, self.N):
            count += self.sample[i].FFn()

        return count/self.N

    def print_sample_table(self):
        print("+ {sign:.2s} + {sign:130s} + {sign:.7s} +".format(sign=("-" * 130)))
        print("| {:^2s} | {:^130s} | {:^7s} |".format("№", "Chromosome", "PSL"))
        print("+ {sign:.2s} + {sign:.130s} + {sign:.7s} +".format(sign=("-" * 130)))
        
        i = 0
        for a in self.sample:
            print("| {:^2d} | {:<130s} | {:^7d} |".format(i, self.sample[i].get_chr(), self.sample[i].PSL()))
            i += 1
        print("+ {sign:.2s} + {sign:.130s} + {sign:.7s} +".format(sign=("-" * 130)))

class GeneticAlgorithm:
    generation_count = 0
    chromosomes = []
    css = []

    def __init__(self):
        self.generation_count = 0
        self.chromosomes = []
        self.css = []

    def genetic_algorithm(self, population):
        cou = 0

        while cou != 3:
            self.generation_count += 1

            population.crossover()
            population.reduction()

            Fss.append(population.Fs())

            if self.generation_count == 3 and cou == 0:
                print("Population 3")
                population.print_sample_table()

            generation = max(population.sample, key=lambda x: x.FFn())

            if generation.PSL() <= PSL:
                if ACF_c(self.chromosomes, generation.chromosome):
                    cou += 1
                    self.chromosomes.append(generation.chromosome)
                    self.css.append(generation.get_chr())

                    if cou == 1:
                        print("Population ", self.generation_count)
                        population.print_sample_table()

                    if cou != 3:
                        Fss.clear()

                self.generation_count = 0
                population = Population(CHR_LEN)

Fss = []

initial_population = Population(CHR_LEN)

Fss.append(initial_population.Fs())

print("Population 0")
initial_population.print_sample_table()

algorithm = GeneticAlgorithm()
algorithm.genetic_algorithm(initial_population)

for cs in algorithm.css:
    print('Code Sequence =', cs)

if args.plot:
    figure0, axis0 = plt.subplots()
    axis0.plot(Fss)
    axis0.set_title("Среднее значение функции приспособленности через поколения")

    figure1, axis1 = plt.subplots()
    axis1.plot(list(range(-(BIT_LEN-1), BIT_LEN)), ACF(algorithm.chromosomes[0]))
    axis1.set_title("АКФ найдённой Code Sequence 1")
    axis1.grid(True)

    figure2, axis2 = plt.subplots()
    axis2.plot(list(range(-(BIT_LEN-1), BIT_LEN)), ACF(algorithm.chromosomes[1]))
    axis2.set_title("АКФ найдённой Code Sequence 2")
    axis2.grid(True)

    figure3, axis3 = plt.subplots()
    axis3.plot(list(range(-(BIT_LEN-1), BIT_LEN)), ACF(algorithm.chromosomes[2]))
    axis3.set_title("АКФ найдённой Code Sequence 3")
    axis3.grid(True)

    plt.show()