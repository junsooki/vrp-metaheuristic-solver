"""
VRP Problem Definitions
- Customer, Depot, VRPInstance classes
- Solution representation
"""

import math
from typing import List


class Customer:
    """Represents a customer in the VRP"""
    def __init__(self, id: int, x: float, y: float, demand: float, 
                 service_time: float = 0.0, tw_start: float = None, tw_end: float = None):
        self.id = id
        self.x = x
        self.y = y
        self.demand = demand
        self.service_time = service_time
        self.time_window_start = tw_start
        self.time_window_end = tw_end
    
    def __repr__(self):
        return f"Customer({self.id}, x={self.x:.1f}, y={self.y:.1f}, demand={self.demand:.1f})"


class Depot:
    """Represents a depot in the VRP"""
    def __init__(self, x: float, y: float, tw_start: float = 0.0, tw_end: float = float('inf')):
        self.x = x
        self.y = y
        self.time_window_start = tw_start
        self.time_window_end = tw_end
    
    def __repr__(self):
        return f"Depot(x={self.x:.1f}, y={self.y:.1f})"


class VRPInstance:
    """VRP problem instance"""
    def __init__(self, name: str, depot: Depot, customers: List[Customer], 
                 vehicle_capacity: float, num_vehicles: int = None):
        self.name = name
        self.depot = depot
        self.customers = customers
        self.vehicle_capacity = vehicle_capacity
        self.num_vehicles = num_vehicles or len(customers)
        self.num_customers = len(customers)
        
        # Precompute distance matrix
        self._distance_matrix = self._compute_distance_matrix()
    
    def _compute_distance_matrix(self):
        """Precompute distance matrix"""
        n = self.num_customers + 1  # +1 for depot
        dist_matrix = {}
        
        # Depot to customers
        for i, customer in enumerate(self.customers):
            dist_matrix[(0, customer.id)] = self._euclidean_distance(self.depot, customer)
            dist_matrix[(customer.id, 0)] = dist_matrix[(0, customer.id)]
        
        # Customer to customer
        for cust_i in self.customers:
            for cust_j in self.customers:
                if cust_i.id != cust_j.id:
                    dist_matrix[(cust_i.id, cust_j.id)] = self._euclidean_distance(cust_i, cust_j)
        
        return dist_matrix
    
    def _euclidean_distance(self, node1, node2):
        """Calculate Euclidean distance"""
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def get_distance(self, from_id: int, to_id: int):
        """Get distance between nodes (0 = depot)"""
        return self._distance_matrix.get((from_id, to_id), 0)
    
    def __repr__(self):
        return f"VRPInstance(name={self.name}, customers={self.num_customers}, capacity={self.vehicle_capacity})"


class Solution:
    """VRP solution representation"""
    def __init__(self, instance: VRPInstance, routes: List[List[int]]):
        self.instance = instance
        self.routes = routes
        self._objective = None
    
    @property
    def objective(self):
        """Calculate and cache objective value"""
        if self._objective is None:
            self._objective = self._calculate_cost()
        return self._objective
    
    def _calculate_cost(self):
        """Calculate total route distance"""
        total_cost = 0.0
        
        for route in self.routes:
            if not route:
                continue
            
            # Depot to first customer
            total_cost += self.instance.get_distance(0, route[0])
            
            # Customer to customer
            for i in range(len(route) - 1):
                total_cost += self.instance.get_distance(route[i], route[i + 1])
            
            # Last customer to depot
            total_cost += self.instance.get_distance(route[-1], 0)
        
        return total_cost
    
    def invalidate_objective(self):
        """Mark objective as needing recomputation"""
        self._objective = None
    
    def is_feasible(self):
        """Check capacity constraints"""
        for route in self.routes:
            total_demand = sum(self.instance.customers[cid - 1].demand for cid in route if cid > 0)
            if total_demand > self.instance.vehicle_capacity:
                return False
        return True
    
    def copy(self):
        """Create deep copy"""
        import copy
        return Solution(self.instance, [route.copy() for route in self.routes])
    
    def __repr__(self):
        return f"Solution(routes={len(self.routes)}, cost={self.objective:.2f}, feasible={self.is_feasible()})"
