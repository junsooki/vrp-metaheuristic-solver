"""
Heuristic class for storing LLM-generated operators
"""

class Heuristic:
    """
    Wrapper class for LLM-generated heuristics
    Stores code, name, and compiled function
    """
    
    def __init__(self, name, code, function):
        """
        Initialize heuristic
        
        Args:
            name: Heuristic name
            code: Source code as string
            function: Compiled callable function
        """
        self.name = name
        self.code = code
        self.function = function
        self.fitness_history = []
    
    def __call__(self, *args, **kwargs):
        """Make heuristic callable"""
        return self.function(*args, **kwargs)
    
    def update_fitness(self, fitness):
        """Record fitness value"""
        self.fitness_history.append(fitness)
    
    def get_average_fitness(self):
        """Get average fitness"""
        if not self.fitness_history:
            return float('inf')
        return sum(self.fitness_history) / len(self.fitness_history)
    
    def __repr__(self):
        return f"Heuristic(name='{self.name}', fitness={self.get_average_fitness():.2f})"

