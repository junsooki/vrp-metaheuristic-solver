"""
VRPAGENT-LNS: Large Neighborhood Search with LLM-Generated Heuristics
Based on Algorithm 1 from the VRPAGENT paper
"""

def LNS_solver(I, f_remove, f_order, max_iterations=1000, acceptance='SA'):
    """
    Large Neighborhood Search solver using LLM-generated operators
    Args:
        I: Problem Instance
        f_remove: Destroy operator (LLM-generated heuristic)
        f_order: Ordering operator for reinsertion
        max_iterations: Maximum number of iterations
        acceptance: Acceptance criterion ('SA' for simulated annealing, 'HC' for hill climbing)
    
    Returns:
        Best solution found
    """
    # Generate initial solution
    s = GenerateStartSolution(I)
    s_best = Copy(s)
    
    iteration = 0
    temperature = InitialTemperature(s)
    
    # Main LNS loop
    while not TerminationCriterionMet(iteration, max_iterations):
        # Destroy: Remove customers using LLM-generated heuristic
        num_remove = max(1, int(0.3 * len(I.customers))) # random removal of 30% of customers
        (s_partial, removed_customers) = f_remove(I, s, num_remove)
        
        # Repair: Reinsert customers in order
        insertion_order = f_order(I, s_partial, removed_customers)
        
        s_new = Copy(s_partial)
        for c in insertion_order:
            s_new = InsertCustomer(I, s_new, c)
        
        # Accept or reject new solution
        s = Accept(s, s_new, temperature, acceptance)
        
        # Update best solution
        if Cost(s) < Cost(s_best):
            s_best = Copy(s)
        
        # Update temperature (for simulated annealing)
        temperature = UpdateTemperature(temperature, iteration)
        
        iteration += 1
    
    return s_best


def GenerateStartSolution(I):
    """ Generate initial solution using construction heuristic """
    # Could use: nearest neighbor, savings algorithm, sweep algorithm, etc.
    return NearestNeighborConstruction(I)


def Accept(s_current, s_new, temperature, method='SA'):
    """Acceptance criterion for new solution"""
    import random
    import math
    
    cost_current = Cost(s_current)
    cost_new = Cost(s_new)
    
    # Always accept better solutions
    if cost_new < cost_current:
        return s_new
    
    # For worse solutions:
    if method == 'SA':
        # Simulated annealing: accept with probability
        delta = cost_new - cost_current
        probability = math.exp(-delta / temperature)
        
        if random.random() < probability:
            return s_new
        else:
            return s_current
    
    elif method == 'HC':
        # Hill climbing: only accept improvements
        return s_current
    
    else:
        # Default: accept worse solutions with fixed probability
        if random.random() < 0.1:
            return s_new
        else:
            return s_current


def InsertCustomer(I, s, customer):
    """
    Insert customer into solution at best position
    
    Args:
        I: Problem instance
        s: Partial solution
        customer: Customer to insert
    
    Returns:
        Solution with customer inserted
    """
    best_cost_increase = float('inf')
    best_route_idx = None
    best_position = None
    
    # Try inserting in each route at each position
    for route_idx, route in enumerate(s.routes):
        for pos in range(len(route) + 1):
            cost_increase = CalculateInsertionCost(I, s, route_idx, pos, customer)
            
            if cost_increase < best_cost_increase:
                # Check feasibility (capacity, time windows, etc.)
                if IsFeasibleInsertion(I, s, route_idx, pos, customer):
                    best_cost_increase = cost_increase
                    best_route_idx = route_idx
                    best_position = pos
    
    # Insert at best position
    if best_route_idx is not None:
        s.routes[best_route_idx].insert(best_position, customer)
    else:
        # Create new route if no feasible insertion found
        s.routes.append([customer])
    
    return s


def CalculateInsertionCost(I, s, route_idx, position, customer):
    """Calculate cost increase of inserting customer at position"""
    route = s.routes[route_idx]
    
    if len(route) == 0:
        # Empty route: depot -> customer -> depot
        return 2 * Distance(I.depot, customer)
    
    # Get neighbors
    if position == 0:
        prev = I.depot
        next_node = route[0]
    elif position == len(route):
        prev = route[-1]
        next_node = I.depot
    else:
        prev = route[position - 1]
        next_node = route[position]
    
    # Cost = new edges - old edge
    old_cost = Distance(prev, next_node)
    new_cost = Distance(prev, customer) + Distance(customer, next_node)
    
    return new_cost - old_cost


def IsFeasibleInsertion(I, s, route_idx, position, customer):
    """Check if insertion maintains feasibility (capacity, time windows)"""
    route = s.routes[route_idx]
    
    # Check capacity constraint
    current_load = sum(c.demand for c in route)
    if current_load + customer.demand > I.vehicle_capacity:
        return False
    
    # Check time windows if applicable
    if hasattr(I, 'time_windows'):
        # Would need to check time window feasibility here
        pass
    
    return True


def InitialTemperature(s):
    """Calculate initial temperature for simulated annealing"""
    return 0.05 * Cost(s)


def UpdateTemperature(temperature, iteration, cooling_rate=0.99):
    """Update temperature for simulated annealing"""
    return temperature * cooling_rate


def TerminationCriterionMet(iteration, max_iterations):
    """Check if LNS should terminate"""
    return iteration >= max_iterations


def Cost(s):
    """Calculate total cost of solution"""
    total_cost = 0
    
    for route in s.routes:
        if len(route) == 0:
            continue
        
        # Depot to first customer
        total_cost += Distance(s.instance.depot, route[0])
        
        # Customer to customer
        for i in range(len(route) - 1):
            total_cost += Distance(route[i], route[i + 1])
        
        # Last customer to depot
        total_cost += Distance(route[-1], s.instance.depot)
    
    return total_cost


def Distance(node1, node2):
    """Calculate Euclidean distance between two nodes"""
    import math
    return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)


def Copy(s):
    """Create deep copy of solution"""
    import copy
    return copy.deepcopy(s)


def NearestNeighborConstruction(I):
    """ Construct initial solution using nearest neighbor heuristic """
    unvisited = set(I.customers)
    routes = []
    
    while unvisited:
        route = []
        current_location = I.depot
        current_capacity = 0
        
        while unvisited:
            # Find nearest feasible customer
            nearest = None
            min_distance = float('inf')
            
            for customer in unvisited:
                if current_capacity + customer.demand <= I.vehicle_capacity:
                    dist = Distance(current_location, customer)
                    if dist < min_distance:
                        min_distance = dist
                        nearest = customer
            
            if nearest is None:
                break  # No feasible customer, start new route
            
            # Add customer to route
            route.append(nearest)
            unvisited.remove(nearest)
            current_location = nearest
            current_capacity += nearest.demand
        
        if route:
            routes.append(route)
    
    # Create solution object
    solution = Solution(instance=I, routes=routes)
    return solution


class Solution:
    """Simple solution representation"""
    def __init__(self, instance, routes):
        self.instance = instance
        self.routes = routes
