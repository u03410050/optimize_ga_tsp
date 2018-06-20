import matplotlib.pyplot as plt
import numpy as np
from io_helper import read_tsp, normalize
from sys import argv

CROSS_RATE = 0.2
MUTATE_RATE = 0.01

N_GENERATIONS = 100


class GA(object):
    def __init__(self, DNA_size, cross_rate, mutation_rate, pop_size, ):
        self.DNA_size = DNA_size
        self.cross_rate = cross_rate
        self.mutate_rate = mutation_rate
        self.pop_size = pop_size

        self.pop = np.vstack([np.random.permutation(DNA_size) for _ in range(pop_size)])

    def translateDNA(self, DNA, city_position):     # get cities' coord in order
        line_x = np.empty_like(DNA, dtype=np.float64)
        line_y = np.empty_like(DNA, dtype=np.float64)
        for i, d in enumerate(DNA):
            city_coord = city_position[d]
            line_x[i, :] = city_coord[:, 0]
            line_y[i, :] = city_coord[:, 1]
        return line_x, line_y

    def get_fitness(self, line_x, line_y):
        total_distance = np.empty((line_x.shape[0],), dtype=np.float64)
        for i, (xs, ys) in enumerate(zip(line_x, line_y)):
            total_distance[i] = np.sum(np.sqrt(np.square(np.diff(xs)) + np.square(np.diff(ys))))
        fitness_mole=0
        if self.DNA_size<500:
            fitness_mole=self.DNA_size*2000
        elif self.DNA_size>=500 and self.DNA_size<5000:
            fitness_mole=self.DNA_size*15000
        elif self.DNA_size>=5000 :
            fitness_mole=self.DNA_size*30000

        fitness = np.exp(fitness_mole/ total_distance)
        return fitness, total_distance

    def select(self, fitness):
        idx = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / fitness.sum())
        #print(fitness / fitness.sum())
        x=np.argsort(fitness / fitness.sum())
        #print("i",idx)
       # print("x1",x)
        x2=x[::-1][:self.pop_size]
      #  print("x2",x2)
        return self.pop[idx]

    def crossover(self, parent, pop):
        rand=np.random.rand()
        if rand < self.cross_rate:
            i_ = np.random.randint(0, self.pop_size, size=1)                        # select another individual from pop
            cross_points = np.random.randint(0, 2, self.DNA_size).astype(np.bool)
            keep_city = parent[~cross_points]
            swap_city = pop[i_, np.isin(pop[i_].ravel(), keep_city, invert=True)]
            parent[:] = np.concatenate((keep_city, swap_city))
        return parent

    def mutate(self, child):
        for point in range(self.DNA_size):
            if np.random.rand() < self.mutate_rate:
                swap_point = np.random.randint(0, self.DNA_size)
                swapA, swapB = child[point], child[swap_point]
                child[point], child[swap_point] = swapB, swapA
        return child

    def evolve(self, fitness):
        pop = self.select(fitness)
        pop_copy = pop.copy()
        i=0
        for parent in pop: 
            if(i>self.pop_size/2):
                break
            child = self.crossover(parent, pop_copy) #pop_copy have 500,n
            child = self.mutate(child)
            parent[:] = child
            i=i+1

        self.pop = pop



class TravelSalesPerson(object):
    def __init__(self, n_cities,x_cities):
        self.city_position = x_cities
        plt.ion()

    def plotting(self, lx, ly, total_d,i):
        plt.cla()
        plt.scatter(self.city_position[:, 0].T, self.city_position[:, 1].T, s=100, c='k')
        plt.plot(lx.T, ly.T, 'r-')
        plt.text(-0.05, -0.05, "Total distance=%.2f" % total_d, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.01)
        plt.savefig("./diagrams/"+str(i)+".png")



def main():
    if len(argv) != 2:
        print("Correct use: python src/main.py <filename>.tsp")
        return -1

    problem = read_tsp(argv[1])
    cities=problem.copy()
    N_CITIES = cities.shape[0]
    x = np.array(cities[['x','y']], np.float64)
    
    POP_SIZE = N_CITIES*10
    ga = GA(DNA_size=N_CITIES, cross_rate=CROSS_RATE, mutation_rate=MUTATE_RATE, pop_size=POP_SIZE)
    env = TravelSalesPerson(n_cities=N_CITIES,x_cities=x)
    
    for generation in range(N_GENERATIONS):
        lx, ly = ga.translateDNA(ga.pop, env.city_position)
        fitness, total_distance = ga.get_fitness(lx, ly)
        ga.evolve(fitness)
        best_idx = np.argmax(fitness)
        if not generation % 50:
            print('Gen:', generation, '| best fit: %.2f' % fitness[best_idx] ,'|total_distance: %.2f' % total_distance[best_idx])
            env.plotting(lx[best_idx], ly[best_idx], total_distance[best_idx],generation)

    plt.ioff()
    plt.show()
    

if __name__=='__main__':
  
    main()