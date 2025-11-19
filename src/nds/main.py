"""
NDS Main Entry Point
Train or run Neural Destroy Selection solver
"""

import argparse
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.problem import Customer, Depot, VRPInstance
from common.data_loader import load_vrp_instance
from nds.trainer import NDSTrainer
from nds.destroy_operator import NeuralDestroyOperator
from nds.search import augmented_simulated_annealing, generate_start_solution, greedy_repair


def train_mode(args):
    """Train NDS model"""
    print("=" * 80)
    print("NDS TRAINING MODE")
    print("=" * 80)
    print()
    
    # Create validation instances if requested
    validation_instances = None
    if args.num_validation > 0:
        print(f"Generating {args.num_validation} validation instances...")
        validation_instances = []
        for i in range(args.num_validation):
            instance = generate_random_instance(
                num_customers=args.instance_size,
                seed=10000 + i
            )
            validation_instances.append(instance)
        print(f" Created {len(validation_instances)} validation instances\n")
    
    # Initialize trainer
    print(f"Initializing trainer...")
    print(f"  Model type: {args.model_type}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Device: {args.device}")
    print()
    
    trainer = NDSTrainer(
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # Train
    trainer.train(
        iterations_per_instance=args.iterations_per_instance,
        rollouts_per_solution=args.rollouts_per_solution,
        improvement_steps=args.improvement_steps,
        num_training_instances=args.num_training,
        instance_size=args.instance_size,
        batch_size=1,
        num_remove_pct=args.removal_percentage,
        grad_accumulation_steps=args.grad_accumulation,
        validation_instances=validation_instances,
        validation_interval=args.validation_interval,
        verbose=True
    )
    
    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, args.model_name)
    trainer.save_model(model_path)
    
    print()
    print("=" * 80)
    print(f"✓ Training complete! Model saved to: {model_path}")
    print("=" * 80)


def solve_mode(args):
    """Solve VRP instance using trained NDS model"""
    print("=" * 80)
    print("NDS SOLVING MODE")
    print("=" * 80)
    print()
    
    # Load instance
    if args.instance_file:
        print(f"Loading instance from: {args.instance_file}")
        instance = load_vrp_instance(args.instance_file)
    else:
        print(f"Generating random instance with {args.instance_size} customers...")
        instance = generate_random_instance(
            num_customers=args.instance_size,
            seed=args.seed
        )
    
    print(f"Instance: {instance}")
    print()
    
    # Load neural destroy operator
    model_path = os.path.join(args.model_dir, args.model_name)
    
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        print("Using untrained model (random policy)")
    
    print(f"Loading neural destroy operator...")
    print(f"  Model: {model_path}")
    print(f"  Temperature: {args.temperature}")
    print()
    
    destroy_op = NeuralDestroyOperator(
        model_path=model_path,
        model_type=args.model_type,
        temperature=args.temperature,
        device=args.device
    )
    
    # Solve using augmented simulated annealing
    print("Running Augmented Simulated Annealing with Neural Destroy...")
    print(f"  Iterations: {args.max_iterations}")
    print(f"  Augmentations: {args.num_augmentations}")
    print(f"  Rollouts: {args.num_rollouts}")
    print()
    
    best_solution = augmented_simulated_annealing(
        instance=instance,
        max_iterations=args.max_iterations,
        num_augmentations=args.num_augmentations,
        num_rollouts=args.num_rollouts,
        lambda_start=args.lambda_start,
        lambda_decay=args.lambda_decay,
        policy_network=None,
        threshold_factor=args.threshold_factor,
        destroy_operator=lambda sol, n: destroy_op(sol, n),
        repair_operator=greedy_repair
    )
    
    # Print results
    print()
    print("=" * 80)
    print("SOLUTION FOUND")
    print("=" * 80)
    print(f"  Total Cost: {best_solution.objective:.2f}")
    print(f"  Number of Routes: {len(best_solution.routes)}")
    print(f"  Feasible: {best_solution.is_feasible()}")
    print()
    print("Routes:")
    for i, route in enumerate(best_solution.routes, 1):
        route_demand = sum(instance.customers[cid - 1].demand for cid in route)
        print(f"  Route {i}: {route} (demand: {route_demand:.2f})")
    print("=" * 80)
    
    # Save solution if requested
    if args.output_file:
        save_solution(best_solution, args.output_file)
        print(f"✓ Solution saved to: {args.output_file}")


def generate_random_instance(num_customers: int, seed: int = None) -> VRPInstance:
    """Generate random VRP instance"""
    if seed is not None:
        np.random.seed(seed)
    
    # Depot at center
    depot = Depot(x=50, y=50)
    
    # Random customers
    customers = []
    for i in range(1, num_customers + 1):
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        demand = np.random.uniform(1, 10)
        customers.append(Customer(id=i, x=x, y=y, demand=demand))
    
    # Vehicle capacity
    capacity = num_customers * 3
    
    instance = VRPInstance(
        name=f"random_{num_customers}",
        depot=depot,
        customers=customers,
        vehicle_capacity=capacity
    )
    
    return instance


def save_solution(solution, output_file):
    """Save solution to file"""
    with open(output_file, 'w') as f:
        f.write(f"Cost: {solution.objective:.2f}\n")
        f.write(f"Routes: {len(solution.routes)}\n")
        f.write(f"Feasible: {solution.is_feasible()}\n")
        f.write("\n")
        for i, route in enumerate(solution.routes, 1):
            f.write(f"Route {i}: {' '.join(map(str, route))}\n")


def main():
    parser = argparse.ArgumentParser(
        description='NDS: Neural Destroy Selection for VRP',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python src/nds/main.py --train --num-training 100 --instance-size 50
  
  # Solve with trained model
  python src/nds/main.py --solve --instance-file data/X-n101-k25.vrp
  
  # Solve random instance
  python src/nds/main.py --solve --instance-size 50 --seed 42
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true', help='Train NDS model')
    mode_group.add_argument('--solve', action='store_true', help='Solve VRP instance')
    
    # Common arguments
    parser.add_argument('--model-type', type=str, default='simple',
                       choices=['simple', 'autoregressive'],
                       help='Neural network architecture (default: simple)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory for model checkpoints (default: models)')
    parser.add_argument('--model-name', type=str, default='nds_model.pt',
                       help='Model filename (default: nds_model.pt)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--num-training', type=int, default=100,
                            help='Number of training instances (default: 100)')
    train_group.add_argument('--num-validation', type=int, default=20,
                            help='Number of validation instances (default: 20)')
    train_group.add_argument('--instance-size', type=int, default=50,
                            help='Number of customers per instance (default: 50)')
    train_group.add_argument('--iterations-per-instance', type=int, default=100,
                            help='Iterations per instance (I in Algorithm 1, default: 100)')
    train_group.add_argument('--rollouts-per-solution', type=int, default=10,
                            help='Rollouts per solution (K in Algorithm 1, default: 10)')
    train_group.add_argument('--improvement-steps', type=int, default=20,
                            help='Improvement steps (J in Algorithm 1, default: 20)')
    train_group.add_argument('--removal-percentage', type=float, default=0.3,
                            help='Percentage of customers to remove (default: 0.3)')
    train_group.add_argument('--learning-rate', type=float, default=1e-4,
                            help='Learning rate (default: 1e-4)')
    train_group.add_argument('--embedding-dim', type=int, default=128,
                            help='Model embedding dimension (default: 128)')
    train_group.add_argument('--num-heads', type=int, default=8,
                            help='Number of attention heads (default: 8)')
    train_group.add_argument('--num-layers', type=int, default=3,
                            help='Number of encoder layers (default: 3)')
    train_group.add_argument('--grad-accumulation', type=int, default=1,
                            help='Gradient accumulation steps (default: 1)')
    train_group.add_argument('--validation-interval', type=int, default=10,
                            help='Validate every N instances (default: 10)')
    
    # Solving arguments
    solve_group = parser.add_argument_group('Solving Options')
    solve_group.add_argument('--instance-file', type=str,
                            help='Path to VRP instance file')
    solve_group.add_argument('--max-iterations', type=int, default=1000,
                            help='Maximum SA iterations (default: 1000)')
    solve_group.add_argument('--num-augmentations', type=int, default=8,
                            help='Number of solution augmentations (default: 8)')
    solve_group.add_argument('--num-rollouts', type=int, default=10,
                            help='Number of rollouts per iteration (default: 10)')
    solve_group.add_argument('--lambda-start', type=float, default=100.0,
                            help='Initial temperature (default: 100.0)')
    solve_group.add_argument('--lambda-decay', type=float, default=0.995,
                            help='Temperature decay rate (default: 0.995)')
    solve_group.add_argument('--threshold-factor', type=float, default=1.0,
                            help='Threshold factor (default: 1.0)')
    solve_group.add_argument('--temperature', type=float, default=1.0,
                            help='Sampling temperature for neural operator (default: 1.0)')
    solve_group.add_argument('--output-file', type=str,
                            help='Output file for solution')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Run appropriate mode
    if args.train:
        train_mode(args)
    elif args.solve:
        solve_mode(args)


if __name__ == "__main__":
    main()

