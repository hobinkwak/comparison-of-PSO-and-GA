## comparison-of-PSO-and-GA
- Particle Swarm Optimization vs. Genetic Algorithm Test

## Test
- Rosenbrock function (=Banana function)
  - minimum solution : (1, 1)
```python
def f(X):
    a, b = 1, 100
    return ((a - X[0]) ** 2) + b*(X[1]-X[0]**2)**2
```

- Rastrigin function
  - minimum solution : (0, 0)
```python
def f(X):
    A = 10
    return A*len(X) + sum([(x**2 - A * np.cos(2 * np.pi * x)) for x in X])
```

## PSO
- Particle Swarm Optimization Code in pso.py
  - Swarm size : **size** (default: 1000, __init__ parameter)
  - Number of iterations : **n_iter** (default: 1000, __init__ parameter)
  - Inertia Weight : **w** (default: 0.5, __init__ parameter)
  - Cognitive Weight : **c1** (default: 0.25, __init__ parameter)
  - Social Weight : **c2** (default: 0.25, __init__ parameter)

## Requirements
- geneticalgorithm==1.0.2
- matplotlib==3.5.1
- numpy==1.20.0
