import numpy as np
from tqdm import tqdm

class PSO:
    def __init__(self, func, dimension, varbound, size=1000, w=0.5, c1=0.25, c2=0.25, n_iter=1000):
        self.f = func
        self.dim = dimension
        self.varbound = varbound
        self.size = size
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.n_iter = n_iter

    def generate(self):
        X = np.array([np.random.uniform(bound[0], bound[-1],
                     size=(self.size,)) for bound in self.varbound])
        V = np.random.randn(self.dim, self.size) * 0.1
        self.X = X  # dim x size
        self.V = V  # dim x size

    def initialize(self):
        pbest_loc = self.X
        pbest_value = self.f(self.X)
        gbest_loc = pbest_loc[:, pbest_value.argmin()]
        gbest_value = pbest_value.min()
        self.pbest_loc = pbest_loc
        self.pbest_value = pbest_value
        self.gbest_loc = gbest_loc
        self.gbest_value = gbest_value

    def update(self):
        r1, r2 = np.random.rand(2)
        self.V = self.w * self.V + r1 * self.c1 * \
            (self.pbest_loc - self.X) + r2 * self.c2 * \
            (self.gbest_loc[..., np.newaxis] - self.X)
        self.X = self.X + self.V
    
    def evaluate(self):
        obj_value = self.f(self.X)
        cond = obj_value <= self.pbest_value
        self.pbest_loc[:, cond] = self.X[:, cond]
        self.pbest_value = np.vstack([self.pbest_value, obj_value]).min(axis=0)
        self.gbest_loc = self.pbest_loc[:, self.pbest_value.argmin()]
        self.gbest_value = self.pbest_value.min()

    def run(self):
        self.generate()
        self.initialize()
        for _ in tqdm(range(self.n_iter)):
            self.update()
            self.evaluate()
        print("Solution:", self.gbest_loc)