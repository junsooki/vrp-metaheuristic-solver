"""
VRPAGENT-GA: Genetic Algorithm for Evolving LLM-Generated Heuristics
Based on Algorithm 2 from the VRPAGENT paper
"""

from Heuristic import Heuristic

def GA(N_init, N_E, N_C) -> Heuristic:
    """
    N_init: Initial population size, N_E: Number of elites, N_C: Number of offspring to generate per generation
    """
    # Initialize population
    P = GenerateStartPop(N_init)
    
    # Evolution loop
    while not TerminationCriteriaMet():
        # Rank heuristics and select top N_E as elites
        (E, NE) = TopKElite(P, N_E)
        
        # Generate offspring through crossover
        C = []
        while len(C) < N_C:
            # Select random elite and non-elite
            p_e = Random(E)
            p_ne = Random(NE)
            
            # Generate new heuristic via biased crossover
            c = BiasedCrossover(p_e, p_ne)
            C.append(c)
        
        # Improve elites via mutation
        for e in E:
            m = Mutation(e)
            # Replace elite if mutated version is better
            if Fit(m) < Fit(e):
                e = m
        
        # Create next generation: elites + offspring
        P = Union(E, C)
    
    # Return best heuristic
    return Best(P)


def GenerateStartPop(N_init):
    """
    Generate initial population of heuristics using LLM
    
    Args:
        N_init: Population size
    
    Returns:
        List of initial heuristics (f_remove operators)
    """
    population = []
    
    for i in range(N_init):
        # Generate new destroy heuristic using LLM Manager
        name, code, compiled_func = LLM.f_generate_llm(
            operator_type='f_remove',
            examples=None,
            temperature=0.9  # High temperature for diversity
        )
        
        if compiled_func is not None:
            # Create heuristic object
            heuristic = Heuristic(name=name, code=code, function=compiled_func)
            population.append(heuristic)
            print(f"Generated heuristic {i+1}/{N_init}: {name}")
        else:
            print(f"Failed to generate heuristic {i+1}/{N_init}")
    
    return population


def TopKElite(P, N_E):
    """ Rank heuristics by fitness and select top N_E as elites"""
    # Evaluate fitness for all heuristics
    fitness_scores = [(h, Fit(h)) for h in P]
    
    # Sort by fitness (lower is better)
    sorted_population = sorted(fitness_scores, key=lambda x: x[1])
    
    # Select elites
    E = [h for h, _ in sorted_population[:N_E]]
    NE = [h for h, _ in sorted_population[N_E:]]
    
    return (E, NE)


def Random(population):
    """ Select random heuristic from population """
    import random
    return random.choice(population)


def Union(E, C):
    """ Combine elites and offspring to create next generation """
    return E + C


def BiasedCrossover(p_e, p_ne):
    """ Generate new heuristic by crossing over elite and non-elite parents Biased towards elite parent """
    # Create crossover prompt with both parents
    prompt = f"""You are an expert in operations research. Create a NEW destroy heuristic by combining these two parent heuristics.

**Elite Parent** (Better Performance - Weight More):
```python
{p_e.code}
```

**Non-Elite Parent**:
```python
{p_ne.code}
```

Create a child heuristic that:
1. Combines the best features from BOTH parents
2. Is biased towards the elite parent's strategy
3. May introduce small variations

Provide:
**Heuristic Name:** [Name]

**Python Code:**
```python
def f_remove(instance, solution, num_remove):
    # Your combined implementation
    pass
```
"""
    
    # Generate child via LLM
    response = LLM.Generate(prompt, temperature=0.7)
    
    try:
        # Parse and compile
        name, code = LLM._parse_llm_response(response)
        compiled_func = LLM._validate_and_compile(code, 'f_remove')
        
        if compiled_func is not None:
            from Heuristic import Heuristic
            child = Heuristic(name=name, code=code, function=compiled_func)
        else:
            print("Crossover compilation failed, returning copy of elite")
            child = Copy(p_e)
    except Exception as ex:
        print(f"Crossover failed: {ex}, returning copy of elite parent")
        child = Copy(p_e)
    
    return child


def Mutation(e):
    """ Modify heuristic using mutation prompt """
    # Create mutation prompt
    mutation_type = SelectMutationType()  # 'parameter', 'logic', 'hybrid'
    
    mutation_instructions = {
        'parameter': 'Adjust numerical parameters (thresholds, probabilities, weights)',
        'logic': 'Modify the selection logic or criteria',
        'hybrid': 'Add or modify subroutines and combine strategies'
    }
    
    prompt = f"""You are an expert in operations research. Create a MUTATED version of this heuristic.

**Original Heuristic:**
```python
{e.code}
```

**Mutation Type:** {mutation_type}
**Instruction:** {mutation_instructions[mutation_type]}

Create a mutated version that:
1. Keeps the core idea
2. Introduces variations based on mutation type
3. Remains a valid f_remove function

Provide:
**Heuristic Name:** [Name]

**Python Code:**
```python
def f_remove(instance, solution, num_remove):
    # Your mutated implementation
    pass
```
"""
    
    # Generate mutated version via LLM
    response = LLM.Generate(prompt, temperature=0.8)
    
    try:
        # Parse and compile
        name, code = LLM._parse_llm_response(response)
        compiled_func = LLM._validate_and_compile(code, 'f_remove')
        
        if compiled_func is not None:
            from Heuristic import Heuristic
            mutated = Heuristic(name=name, code=code, function=compiled_func)
        else:
            print("Mutation compilation failed, returning copy")
            mutated = Copy(e)
    except Exception as ex:
        print(f"Mutation failed: {ex}, returning copy of original heuristic")
        mutated = Copy(e)
    
    return mutated


def Fit(h, lambda_penalty=0.01):
    """
    Evaluate fitness of heuristic according to paper formula:
    Fit(i) = (1/|I^train|) * Σ Obj(s_{i,j}) + λ · Len(C_i)
    
    Obj(s_{i,j}) is the objective value (cost) of solution using heuristic i on instance j
    Len(C_i) is the code length (complexity) of heuristic i
    λ is the penalty weight for code complexity
    """
    total_objective = 0
    num_instances = len(EvaluationInstances)
    from importlib import import_module
    lns_module = import_module('vrpagent.VRPAgent-LNS')
    LNS_solver = lns_module.LNS_solver
    
    for instance in EvaluationInstances:
        try:
            # Run LNS with this heuristic
            s_final = LNS_solver(
                I=instance,
                f_remove=h.function,  # Use this heuristic as destroy operator
                f_order=DefaultOrderingOperator
            )
            
            # Get objective value (solution cost)
            obj_value = Cost(s_final)
            total_objective += obj_value
        except Exception as e:
            # If heuristic fails during execution, assign very poor fitness
            print(f"Warning: Heuristic {h.name if hasattr(h, 'name') else 'unknown'} failed on instance: {e}")
            # Add penalty equal to a very poor solution (e.g., 10x the worst expected cost)
            total_objective += 10000  # Large penalty for failed heuristics
    
    # Average objective value: (1/|I^train|) * Σ Obj(s_{i,j})
    avg_objective = total_objective / num_instances
    
    # Code length penalty: λ · Len(C_i)
    code_length = Len(h)
    complexity_penalty = lambda_penalty * code_length
    
    # Fitness = average objective + complexity penalty
    fitness = avg_objective + complexity_penalty
    
    return fitness


def Best(P):
    """
    Return best heuristic from population
    
    Args:
        P: Population
    
    Returns:
        Best heuristic (lowest fitness)
    """
    best_heuristic = None
    best_fitness = float('inf')
    
    for h in P:
        fitness = Fit(h)
        if fitness < best_fitness:
            best_fitness = fitness
            best_heuristic = h
    
    return best_heuristic


def TerminationCriteriaMet():
    """
    Check if GA should terminate
    
    Returns:
        True if should terminate, False otherwise
    """
    # Check various termination criteria:
    # - Maximum generations reached
    # - No improvement for N generations
    # - Fitness threshold reached
    # - Time limit exceeded
    
    if CurrentGeneration >= MaxGenerations:
        return True
    
    if GenerationsWithoutImprovement >= PatienceLimit:
        return True
    
    if BestFitness < TargetFitness:
        return True
    
    if ElapsedTime >= TimeLimit:
        return True
    
    return False


def SelectMutationType():
    """Randomly select mutation type"""
    import random
    return random.choice(['parameter', 'logic', 'hybrid'])


def Copy(heuristic):
    """Create copy of heuristic"""
    import copy
    return copy.deepcopy(heuristic)


def Cost(solution):
    """ Calculate solution cost (total route distance) """
    total_cost = 0
    
    for route in solution.routes:
        if len(route) == 0:
            continue
        
        # Depot to first customer
        total_cost += Distance(solution.instance.depot, route[0])
        
        # Customer to customer
        for i in range(len(route) - 1):
            total_cost += Distance(route[i], route[i + 1])
        
        # Last customer to depot
        total_cost += Distance(route[-1], solution.instance.depot)
    
    return total_cost


def Distance(node1, node2):
    """
    Calculate Euclidean distance between nodes      
    node1, node2: Nodes with x, y coordinates
    """
    import math
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def Len(h):
    """ Calculate code length (complexity) of heuristic """
    if hasattr(h, 'code'):
        # Count lines of code (excluding comments and blank lines)
        lines = h.code.split('\n')
        code_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        return len(code_lines)
    else:
        # Default complexity
        return 100


# Global variables (would be set by main program)
LLM = None
EvaluationInstances = []
CurrentGeneration = 0
MaxGenerations = 50
GenerationsWithoutImprovement = 0
PatienceLimit = 10
BestFitness = float('inf')
TargetFitness = -10.0  # Target 10% improvement
TimeLimit = 3600  # 1 hour
ElapsedTime = 0
DefaultOrderingOperator = None

