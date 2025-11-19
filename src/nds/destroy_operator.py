"""
Neural Destroy Operator
Wrapper for using trained neural network as destroy operator in LNS
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.problem import VRPInstance, Solution
from nds.model import create_model


class NeuralDestroyOperator:
    """
    Neural destroy operator using trained policy network
    
    Uses a trained neural network to select which customers to remove
    from the current solution. Compatible with search.py interface.
    """
    
    def __init__(self, device: str = 'cuda', model_path: str = None, model_type: str = 'simple', embedding_dim: int = 128, num_heads: int = 8, num_layers: int = 3, dropout: float = 0.1, temperature: float = 1.0,):
        """
        Initialize neural destroy operator
        
        Args:
            model_path: Path to trained model checkpoint (if None, uses untrained model)
            model_type: 'simple' or 'autoregressive'
            embedding_dim: Model embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            device: 'cpu' or 'cuda'
            temperature: Sampling temperature (higher = more exploration)
        """
        self.device = torch.device(device)
        self.temperature = temperature
        
        # Create model
        self.model = create_model(model_type=model_type, embedding_dim=embedding_dim, num_heads=num_heads, num_layers=num_layers).to(self.device)
        
        # Load trained weights if provided
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained model from {model_path}")
        else:
            if model_path:
                print(f"Warning: Model path '{model_path}' not found. Using untrained model.")
            else:
                print("Using untrained model (random policy)")
        
        self.model.eval()  # Set to evaluation mode
    
    def __call__(self, solution: Solution, num_remove: int = None ) -> Tuple[Solution, List[int]]:
        """
        Remove customers from solution using neural policy
        
        Args:
            solution: Current solution
            num_remove: Number of customers to remove (if None, uses 30%)
        
        Returns:
            Tuple of (partial_solution, removed_customers)
        """
        instance = solution.instance
        
        # Determine number of customers to remove
        if num_remove is None:
            num_remove = max(1, int(0.3 * instance.num_customers))
        
        # Select customers using neural network
        removed_customer_ids = self._select_customers(instance, solution, num_remove)
        
        # Remove customers from solution
        partial_solution = solution.copy()
        for customer_id in removed_customer_ids:
            for route in partial_solution.routes:
                if customer_id in route:
                    route.remove(customer_id)
                    break
        
        # Remove empty routes
        partial_solution.routes = [r for r in partial_solution.routes if r]
        partial_solution.invalidate_objective()
        
        return partial_solution, removed_customer_ids
    
    def _select_customers(self, instance: VRPInstance, solution: Solution, num_remove: int) -> List[int]:
        """ Use neural network to select which customers to remove """
        with torch.no_grad():
            # Prepare input tensors
            depot_xy = torch.tensor(
                [[instance.depot.x, instance.depot.y]],
                dtype=torch.float32,
                device=self.device
            )
            
            node_xy = torch.tensor(
                [[c.x, c.y] for c in instance.customers],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            node_demand = torch.tensor(
                [c.demand for c in instance.customers],
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)
            
            # Run model
            selected_indices, log_probs = self.model(
                depot_xy,
                node_xy,
                node_demand,
                num_remove,
                temperature=self.temperature
            )
            
            # Convert from model indices (0-based) to customer IDs (1-based)
            customer_ids = (selected_indices[0] + 1).cpu().tolist()
            
            return customer_ids
    
    def set_temperature(self, temperature: float):
        """Update sampling temperature"""
        self.temperature = temperature