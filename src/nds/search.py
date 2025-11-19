"""
Augmented Simulated Annealing Search Algorithm
Implements Algorithm 2: Augmented Simulated Annealing with Neural Destroy Selection
"""

import copy
import math
import random
from typing import List, Tuple, Callable, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.problem import VRPInstance, Solution


def augmented_simulated_annealing(
    instance: VRPInstance, 
    max_iterations: int, 
    num_augmentations: int, 
    num_rollouts: int, 
    lambda_start: float, 
    lambda_decay: float, 
    policy_network: Optional[Callable] = None, 
    threshold_factor: float = 1.0,  
    destroy_operator: Optional[Callable] = None,
    repair_operator: Optional[Callable] = None) -> Solution:
    """ Augmented Simulated Annealing (Algorithm 2) with Neural Destroy Selection """
    # Initialize augmentations (step 3)
    augmented_instances = create_augmentations(instance, num_augmentations)
    
    # Generate start solutions for each augmentation (step 4)
    solutions = []
    for aug_instance in augmented_instances:
        s = generate_start_solution(aug_instance)
        solutions.append(s)
    
    # Initialize temperature (step 2)
    temperature = lambda_start
    
    # Main search loop (step 5)
    for iteration in range(max_iterations):
        # Improvement step for each augmentation (step 6)
        for a in range(num_augmentations):
            solutions[a] = improvement_step(
                solutions[a],
                policy_network,
                temperature,
                num_rollouts,
                destroy_operator,
                repair_operator
            )
        
        # Calculate costs for all solutions (step 7)
        costs = [solution.objective for solution in solutions]
        
        # Find minimum cost (step 8)
        cost_star = min(costs)
        
        # Calculate threshold (step 9)
        threshold = cost_star + (temperature * threshold_factor)
        
        # Replace poor solutions (steps 10-13)
        for a in range(num_augmentations):
            if costs[a] > threshold:
                # Get solutions better than threshold
                better_solutions = [
                    solutions[i] for i in range(num_augmentations)
                    if costs[i] < threshold
                ]
                
                if better_solutions:
                    # Random choice from better solutions
                    solutions[a] = random.choice(better_solutions).copy()
        
        # Reduce temperature (step 15)
        temperature = reduce_temperature(temperature, lambda_decay)
        
        # Optional: Print progress
        if (iteration + 1) % 100 == 0:
            best_cost = min(solution.objective for solution in solutions)
            avg_cost = sum(solution.objective for solution in solutions) / len(solutions)
            print(f"Iteration {iteration + 1}/{max_iterations}: "
                  f"Best={best_cost:.2f}, Avg={avg_cost:.2f}, Temp={temperature:.4f}")
    
    # Return best solution
    best_solution = min(solutions, key=lambda s: s.objective)
    return best_solution


def create_augmentations(instance: VRPInstance, num_augmentations: int) -> List[VRPInstance]:
    """
    Create augmented instances (e.g., perturbed versions of the original)
    
    Args:
        instance: Original VRP instance
        num_augmentations: Number of augmentations to create
    
    Returns:
        List of augmented instances
    """
    augmented_instances = []
    
    for _ in range(num_augmentations):
        # For now, return copies of the same instance
        # In practice, you might want to add noise to coordinates, demands, etc.
        augmented_instances.append(instance)
    
    return augmented_instances


def generate_start_solution(instance: VRPInstance) -> Solution:
    """
    Generate initial solution using nearest neighbor construction
    
    Args:
        instance: VRP instance
    
    Returns:
        Initial solution
    """
    unvisited = set(range(1, instance.num_customers + 1))  # Customer IDs
    routes = []
    
    while unvisited:
        route = []
        current_location_id = 0  # Start at depot
        current_capacity = 0
        
        while unvisited:
            # Find nearest feasible customer
            nearest_id = None
            min_distance = float('inf')
            
            for customer_id in unvisited:
                customer = instance.customers[customer_id - 1]
                if current_capacity + customer.demand <= instance.vehicle_capacity:
                    dist = instance.get_distance(current_location_id, customer_id)
                    if dist < min_distance:
                        min_distance = dist
                        nearest_id = customer_id
            
            if nearest_id is None:
                break  # No feasible customer, start new route
            
            # Add customer to route
            customer = instance.customers[nearest_id - 1]
            route.append(nearest_id)
            unvisited.remove(nearest_id)
            current_location_id = nearest_id
            current_capacity += customer.demand
        
        if route:
            routes.append(route)
    
    return Solution(instance=instance, routes=routes)


def improvement_step(
    solution: Solution,
    policy_network: Optional[Callable],
    temperature: float,
    num_rollouts: int,
    destroy_operator: Optional[Callable] = None,
    repair_operator: Optional[Callable] = None
) -> Solution:
    """
    Perform one improvement step using LNS
    
    Args:
        solution: Current solution
        policy_network: Neural policy for destroy operator selection
        temperature: Current temperature
        num_rollouts: Number of rollouts (K)
    
    Returns:
        Improved solution (or original if no improvement)
    """
    best_solution = solution
    best_cost = solution.objective
    
    for _ in range(num_rollouts):
        # Destroy phase
        if destroy_operator:
            partial_solution, removed_customers = destroy_operator(solution)
        else:
            # Default: random removal
            num_remove = max(1, int(0.3 * solution.instance.num_customers))
            partial_solution, removed_customers = random_destroy(solution, num_remove)
        
        # Repair phase
        if repair_operator:
            new_solution = repair_operator(partial_solution, removed_customers)
        else:
            # Default: greedy repair
            new_solution = greedy_repair(partial_solution, removed_customers)
        
        # Accept or reject using simulated annealing
        if accept_solution(solution, new_solution, temperature):
            solution = new_solution
            
            # Update best if improved
            if new_solution.objective < best_cost:
                best_solution = new_solution
                best_cost = new_solution.objective
    
    return best_solution


def random_destroy(solution: Solution, num_remove: int) -> Tuple[Solution, List[int]]:
    """ Randomly remove customers from solution """
    partial_solution = solution.copy()
    removed_customers = []
    
    # Get all customer IDs
    all_customers = [cid for route in partial_solution.routes for cid in route]
    
    if not all_customers:
        return partial_solution, removed_customers
    
    # Select customers to remove
    num_to_remove = min(num_remove, len(all_customers))
    customers_to_remove = random.sample(all_customers, num_to_remove)
    
    # Remove from routes
    for customer_id in customers_to_remove:
        for route in partial_solution.routes:
            if customer_id in route:
                route.remove(customer_id)
                removed_customers.append(customer_id)
                break
    
    # Remove empty routes
    partial_solution.routes = [r for r in partial_solution.routes if r]
    partial_solution.invalidate_objective()
    
    return partial_solution, removed_customers


def greedy_repair(partial_solution: Solution, removed_customers: List[int]) -> Solution:
    """ Greedily reinsert removed customers """
    solution = partial_solution.copy()
    instance = solution.instance
    
    for customer_id in removed_customers:
        customer = instance.customers[customer_id - 1]
        
        best_cost_increase = float('inf')
        best_route_idx = None
        best_position = None
        
        # Try inserting in each route at each position
        for route_idx, route in enumerate(solution.routes):
            # Check capacity feasibility
            route_demand = sum(instance.customers[cid - 1].demand for cid in route)
            if route_demand + customer.demand <= instance.vehicle_capacity:
                # Try each position in the route
                for pos in range(len(route) + 1):
                    cost_increase = calculate_insertion_cost(instance, route, pos, customer_id)
                    
                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_route_idx = route_idx
                        best_position = pos
        
        # Try creating new route
        new_route_cost = 2 * instance.get_distance(0, customer_id)
        if new_route_cost < best_cost_increase:
            solution.routes.append([customer_id])
        elif best_route_idx is not None:
            solution.routes[best_route_idx].insert(best_position, customer_id)
        else:
            # No feasible insertion found, create new route anyway
            solution.routes.append([customer_id])
    
    solution.invalidate_objective()
    return solution


def calculate_insertion_cost(
    instance: VRPInstance,
    route: List[int],
    position: int,
    customer_id: int
) -> float:
    """ Calculate cost increase of inserting customer at position in route """
    if not route:
        # Empty route: depot -> customer -> depot
        return 2 * instance.get_distance(0, customer_id)
    
    # Get neighbors
    if position == 0:
        prev_id = 0  # depot
        next_id = route[0]
    elif position == len(route):
        prev_id = route[-1]
        next_id = 0  # depot
    else:
        prev_id = route[position - 1]
        next_id = route[position]
    
    # Calculate cost change
    old_cost = instance.get_distance(prev_id, next_id)
    new_cost = instance.get_distance(prev_id, customer_id) + instance.get_distance(customer_id, next_id)
    
    return new_cost - old_cost


def accept_solution(current: Solution, new: Solution, temperature: float) -> bool:
    """ Simulated annealing acceptance criterion """
    current_cost = current.objective
    new_cost = new.objective
    
    # Always accept better solutions
    if new_cost < current_cost:
        return True
    
    # Accept worse solutions with probability exp(-Δ/T)
    if temperature > 0:
        delta = new_cost - current_cost
        probability = math.exp(-delta / temperature)
        return random.random() < probability
    
    return False


def reduce_temperature(temperature: float, decay_rate: float) -> float:
    """ Reduce temperature using decay rate """
    return temperature * decay_rate

