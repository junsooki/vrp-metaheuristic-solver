"""
NDS Training Loop
Implements Algorithm 1: NDS Training from the paper
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Callable
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.problem import VRPInstance, Solution, Customer, Depot
from nds.model import create_model
from nds.search import (
    augmented_simulated_annealing,
    generate_start_solution,
    greedy_repair,
    random_destroy
)


class NDSTrainer:
    """
    Trainer for Neural Destroy Selection
    Follows Algorithm 1 from the paper
    """
    
    def __init__( self, model_type: str = 'simple', embedding_dim: int = 128, num_heads: int = 8, num_layers: int = 3, learning_rate: float = 1e-4, device: str = 'cpu'):
        """ Initialize trainer """
        self.device = torch.device(device)
        
        # Initialize policy network π_θ (step 2 of Algorithm 1)
        self.model = create_model(
            model_type=model_type,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training state
        self.best_val_score = float('inf')
        self.training_history = {
            'train_rewards': [],
            'train_costs': [],
            'val_costs': []
        }
        
        print(f"Initialized NDS Trainer with {model_type} model")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self, iterations_per_instance: int, rollouts_per_solution: int, improvement_steps: int, num_training_instances: int = 100, instance_size: int = 50, batch_size: int = 8, num_remove_pct: float = 0.3, grad_accumulation_steps: int = 1, validation_instances: List[VRPInstance] = None, validation_interval: int = 10, verbose: bool = True):
        """
        Training procedure following Algorithm 1
        
        Args:
            iterations_per_instance: I - iterations per instance
            rollouts_per_solution: K - rollouts per solution
            improvement_steps: J - improvement steps per instance
            num_training_instances: Number of instances to train on
            instance_size: Number of customers per instance
            batch_size: Batch size for training
            num_remove_pct: Percentage of customers to remove (e.g., 0.3 = 30%)
            grad_accumulation_steps: Accumulate gradients over N steps before updating (default=1, no accumulation)
            validation_instances: List of instances for validation (optional)
            validation_interval: Run validation every N instances (default=10)
            verbose: Print progress
        """
        print("=" * 80)
        print("Starting NDS Training (Algorithm 1)")
        print(f"  Training instances: {num_training_instances}")
        print(f"  Instance size: {instance_size} customers")
        print(f"  Iterations per instance: {iterations_per_instance}")
        print(f"  Rollouts per solution: {rollouts_per_solution}")
        print(f"  Improvement steps: {improvement_steps}")
        print(f"  Gradient accumulation steps: {grad_accumulation_steps}")
        if validation_instances:
            print(f"  Validation instances: {len(validation_instances)}")
        print("=" * 80)
        print()
        
        iteration = 0
        instance_count = 0
        total_iterations = num_training_instances * iterations_per_instance
        grad_acc_counter = 0
        
        # Zero gradients before starting
        self.optimizer.zero_grad()
        
        # Training loop (step 3 of Algorithm 1)
        while iteration < total_iterations:
            # Generate instance (step 4)
            instance = self._generate_instance(instance_size)
            
            # Generate start solution (step 5)
            solution = generate_start_solution(instance)
            
            if verbose and iteration % 10 == 0:
                print(f"Iteration {iteration}/{total_iterations}: Initial cost = {solution.objective:.2f}")
            
            # Iterate on this instance (step 6-18)
            for j in range(improvement_steps):
                # Improvement step using Algorithm 2 (step 7)
                solution, reward, cost = self._improvement_step_with_policy(
                    instance,
                    solution,
                    rollouts_per_solution,
                    num_remove_pct,
                    grad_accumulation_steps
                )
                
                # Track metrics
                self.training_history['train_rewards'].append(reward)
                self.training_history['train_costs'].append(cost)
                
                grad_acc_counter += 1
                iteration += 1
                
                # Optimizer step after accumulating enough gradients
                if grad_acc_counter % grad_accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                if iteration >= total_iterations:
                    break
            
            instance_count += 1
            
            if verbose and iteration % 10 == 0:
                avg_reward = np.mean(self.training_history['train_rewards'][-10:])
                avg_cost = np.mean(self.training_history['train_costs'][-10:])
                print(f"  Iteration {iteration}/{total_iterations}: cost={avg_cost:.2f}, reward={avg_reward:.4f}")
            
            # Validation (periodic)
            if validation_instances and instance_count % validation_interval == 0:
                val_cost = self.validate(validation_instances, num_remove_pct, verbose=verbose)
                self.training_history['val_costs'].append(val_cost)
                
                # Save best model
                if val_cost < self.best_val_score:
                    self.best_val_score = val_cost
                    if verbose:
                        print(f"  ✓ New best validation score: {val_cost:.2f}")
                        self.save_model('best_model.pt')
        
        print("=" * 80)
        print("Training Complete!")
        print("=" * 80)
    
    def validate(self, validation_instances: List[VRPInstance], num_remove_pct: float = 0.3, num_rollouts: int = 10, verbose: bool = True) -> float:
        """
        Validate model on held-out instances
        
        Args:
            validation_instances: List of validation instances
            num_remove_pct: Percentage of customers to remove
            num_rollouts: Number of rollouts for validation
            verbose: Print progress
        
        Returns:
            Average final cost across validation instances
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Running Validation...")
        
        self.model.eval()  # Set to evaluation mode
        
        all_costs = []
        with torch.no_grad():  # No gradients during validation
            for i, instance in enumerate(validation_instances):
                # Generate initial solution
                solution = generate_start_solution(instance)
                initial_cost = solution.objective
                
                # Run improvement with neural policy
                for _ in range(10):  # 10 improvement iterations
                    best_solution = solution
                    best_cost = solution.objective
                    
                    # Try multiple rollouts
                    for _ in range(num_rollouts):
                        num_remove = max(1, int(num_remove_pct * instance.num_customers))
                        
                        # Use policy to select customers
                        removed_customers, _ = self._rollout_policy(
                            instance,
                            solution,
                            num_remove,
                            temperature=0.5  # Lower temperature for more greedy selection
                        )
                        
                        # Remove and repair
                        partial_solution = self._remove_customers(solution, removed_customers)
                        new_solution = greedy_repair(partial_solution, removed_customers)
                        
                        # Keep best
                        if new_solution.objective < best_cost:
                            best_solution = new_solution
                            best_cost = new_solution.objective
                    
                    solution = best_solution
                
                final_cost = solution.objective
                improvement = ((initial_cost - final_cost) / initial_cost) * 100
                all_costs.append(final_cost)
                
                if verbose:
                    print(f"  Instance {i+1}/{len(validation_instances)}: "
                          f"Initial={initial_cost:.2f}, Final={final_cost:.2f}, "
                          f"Improvement={improvement:.2f}%")
        
        avg_cost = np.mean(all_costs)
        if verbose:
            print(f"  Average validation cost: {avg_cost:.2f}")
            print("=" * 60 + "\n")
        
        self.model.train()  # Back to training mode
        return avg_cost
    
    def _improvement_step_with_policy(self, instance: VRPInstance, solution: Solution, num_rollouts: int, num_remove_pct: float, grad_accumulation_steps: int = 1) -> Tuple[Solution, float, float]:
        """
        Single improvement step using neural policy (corresponds to ImprovementStep in Algorithm 1)
        
        Steps 9-17 of Algorithm 1:
        - Sample K rollouts using policy π_θ
        - For each rollout: remove customers, reinsert greedily
        - Calculate rewards (improvement)
        - Compute gradients and update policy
        
        Args:
            instance: VRP instance
            solution: Current solution
            num_rollouts: K rollouts
            num_remove_pct: Percentage of customers to remove
        
        Returns:
            Tuple of (improved_solution, best_reward, best_cost)
        """
        num_remove = max(1, int(num_remove_pct * instance.num_customers))
        
        # Step 9: Initialize lists for rollouts
        all_removed_customers = []  # τ_k for each rollout
        all_partial_solutions = []  # s̃_k for each rollout
        all_new_solutions = []      # s'_k for each rollout
        all_log_probs = []          # log π_θ(τ_k | ...) for each rollout
        
        # Step 10: Sample K rollouts using policy π_θ
        for k in range(num_rollouts):
            # Use policy network to select which customers to remove
            removed_customers, log_probs = self._rollout_policy(
                instance,
                solution,
                num_remove
            )
            
            # Step 11: Remove customers (corresponds to RemoveCustomers)
            partial_solution = self._remove_customers(solution, removed_customers)
            
            # Step 12: Greedy insertion (corresponds to GreedyInsertion)
            new_solution = greedy_repair(partial_solution, removed_customers)
            
            all_removed_customers.append(removed_customers)
            all_partial_solutions.append(partial_solution)
            all_new_solutions.append(new_solution)
            all_log_probs.append(log_probs)
        
        # Step 13: Calculate rewards r_k = max(Obj(s) - Obj(s'_k), 0)
        rewards = []
        old_cost = solution.objective
        for new_solution in all_new_solutions:
            new_cost = new_solution.objective
            reward = max(old_cost - new_cost, 0)  # Improvement (positive is better)
            rewards.append(reward)
        
        rewards = torch.tensor(rewards, device=self.device)
        
        # Step 14: Calculate baseline b = (1/K) Σ r_k
        baseline = rewards.mean()
        
        # Step 15: Find best rollout k* = arg max r_k
        best_k = rewards.argmax().item()
        
        # Step 16: Calculate gradients g_i = (r_k* - b) ∇_θ log π_θ(τ_k* | ...)
        # This is REINFORCE with baseline
        advantage = rewards[best_k] - baseline
        
        # Get log probability of best rollout
        best_log_probs = all_log_probs[best_k]
        
        # Policy gradient loss (negative because we want to maximize reward)
        # Scale by grad_accumulation_steps for proper averaging
        loss = -(advantage * best_log_probs.sum()) / grad_accumulation_steps
        
        # Backprop (accumulate gradients)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Note: optimizer.step() is called outside, after accumulating multiple gradients
        # For grad_accumulation_steps=1, this behaves as normal
        
        # Step 17: Update solution with best found
        solution = all_new_solutions[best_k]
        best_reward = rewards[best_k].item()
        best_cost = solution.objective
        
        return solution, best_reward, best_cost
    
    def _rollout_policy(self, instance: VRPInstance, solution: Solution, num_remove: int, temperature: float = 1.0) -> Tuple[List[int], torch.Tensor]:
        """
        Use policy network π_θ to select customers to remove
        
        Args:
            instance: VRP instance
            solution: Current solution
            num_remove: Number of customers to remove
            temperature: Sampling temperature
        
        Returns:
            Tuple of (removed_customer_ids, log_probabilities)
        """
        # Prepare input tensors
        depot_xy = torch.tensor([[instance.depot.x, instance.depot.y]], 
                                dtype=torch.float32, device=self.device)
        
        node_xy = torch.tensor([[c.x, c.y] for c in instance.customers],
                              dtype=torch.float32, device=self.device).unsqueeze(0)
        
        node_demand = torch.tensor([c.demand for c in instance.customers],
                                   dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Run model
        self.model.train()
        selected_indices, log_probs = self.model(
            depot_xy,
            node_xy,
            node_demand,
            num_remove,
            temperature
        )
        
        # Convert from model indices (0-based, excluding depot) to customer IDs (1-based)
        # Model output: indices 0, 1, 2, ... correspond to customers with IDs 1, 2, 3, ...
        customer_ids = (selected_indices[0] + 1).cpu().tolist()  # +1 because customer IDs start at 1
        
        return customer_ids, log_probs[0]  # Return first (and only) batch element
    
    def _remove_customers(self, solution: Solution, removed_customer_ids: List[int]) -> Solution:
        """ Remove specified customers from solution """
        partial_solution = solution.copy()
        
        # Remove customers from routes
        for customer_id in removed_customer_ids:
            for route in partial_solution.routes:
                if customer_id in route:
                    route.remove(customer_id)
                    break
        
        # Remove empty routes
        partial_solution.routes = [r for r in partial_solution.routes if r]
        partial_solution.invalidate_objective()
        
        return partial_solution
    
    def _generate_instance(self, num_customers: int, seed: int = None) -> VRPInstance:
        """
        Generate random VRP instance for training
        
        Args:
            num_customers: Number of customers
            seed: Random seed (optional)
        
        Returns:
            VRP instance
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate depot at center
        depot = Depot(x=50, y=50)
        
        # Generate random customers
        customers = []
        for i in range(1, num_customers + 1):
            x = np.random.uniform(0, 100)
            y = np.random.uniform(0, 100)
            demand = np.random.uniform(1, 10)
            customers.append(Customer(id=i, x=x, y=y, demand=demand))
        
        # Vehicle capacity
        capacity = num_customers * 3  # Generous capacity
        
        instance = VRPInstance(
            name=f"train_{num_customers}",
            depot=depot,
            customers=customers,
            vehicle_capacity=capacity
        )
        
        return instance
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")