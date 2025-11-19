"""
Main Entry Point for VRP Solver
Choose between VRPAgent (LLM-based) and NDS (Neural Destroy Selection)
"""

import argparse
import sys
import subprocess


def run_vrpagent(args):
    """Run VRPAgent (LLM-based heuristic generation)"""
    print("=" * 80)
    print("Running VRPAGENT (LLM-based Heuristic Evolution)")
    print("=" * 80)
    print()
    
    # Build command for VRPAgent
    cmd = ['python3', 'src/vrpagent/main.py']
    
    # Forward relevant arguments
    if hasattr(args, 'config') and args.config:
        cmd.extend(['--config', args.config])
    
    if hasattr(args, 'instances') and args.instances:
        cmd.extend(['--instances', args.instances])
    
    if hasattr(args, 'output_dir') and args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    
    # Add any remaining arguments
    if args.extra_args:
        cmd.extend(args.extra_args)
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run VRPAgent
    result = subprocess.run(cmd)
    return result.returncode


def run_nds(args):
    """Run NDS (Neural Destroy Selection)"""
    print("=" * 80)
    print("Running NDS (Neural Destroy Selection)")
    print("=" * 80)
    print()
    
    # Build command for NDS
    cmd = ['python3', 'src/nds/main.py']
    
    # Mode selection
    if args.train:
        cmd.append('--train')
        print("Mode: TRAINING")
    else:
        cmd.append('--solve')
        print("Mode: SOLVING")
    
    print()
    
    # Forward NDS-specific arguments
    if hasattr(args, 'model_type') and args.model_type:
        cmd.extend(['--model-type', args.model_type])
    
    if hasattr(args, 'model_dir') and args.model_dir:
        cmd.extend(['--model-dir', args.model_dir])
    
    if hasattr(args, 'model_name') and args.model_name:
        cmd.extend(['--model-name', args.model_name])
    
    if hasattr(args, 'device') and args.device:
        cmd.extend(['--device', args.device])
    
    # Training-specific
    if args.train:
        if hasattr(args, 'num_training') and args.num_training:
            cmd.extend(['--num-training', str(args.num_training)])
        
        if hasattr(args, 'instance_size') and args.instance_size:
            cmd.extend(['--instance-size', str(args.instance_size)])
        
        if hasattr(args, 'num_validation') and args.num_validation:
            cmd.extend(['--num-validation', str(args.num_validation)])
    
    # Solving-specific
    else:
        if hasattr(args, 'instance_file') and args.instance_file:
            cmd.extend(['--instance-file', args.instance_file])
        
        if hasattr(args, 'instance_size') and args.instance_size:
            cmd.extend(['--instance-size', str(args.instance_size)])
        
        if hasattr(args, 'max_iterations') and args.max_iterations:
            cmd.extend(['--max-iterations', str(args.max_iterations)])
        
        if hasattr(args, 'output_file') and args.output_file:
            cmd.extend(['--output-file', args.output_file])
    
    # Add any remaining arguments
    if args.extra_args:
        cmd.extend(args.extra_args)
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run NDS
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description='VRP Solver: Choose between VRPAgent (LLM) or NDS (Neural)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VRPAgent: Generate and evolve heuristics using LLM
  python src/main.py --agent vrpagent --config config.yaml
  
  # NDS: Train neural destroy selection model
  python src/main.py --agent nds --train --num-training 100
  
  # NDS: Solve instance with trained model
  python src/main.py --agent nds --instance-file data/instance.vrp
  
  # Get help for specific agent
  python src/nds/main.py --help
  python src/vrpagent/main.py --help
        """
    )
    
    # Agent selection (required)
    parser.add_argument('--agent', type=str, required=True,
                       choices=['vrpagent', 'nds'],
                       help='Choose solver: vrpagent (LLM-based) or nds (neural)')
    
    # NDS-specific arguments
    nds_group = parser.add_argument_group('NDS Options')
    nds_group.add_argument('--train', action='store_true',
                          help='Train NDS model (otherwise solve)')
    nds_group.add_argument('--model-type', type=str,
                          choices=['simple', 'autoregressive'],
                          help='Neural network architecture')
    nds_group.add_argument('--model-dir', type=str,
                          help='Directory for model checkpoints')
    nds_group.add_argument('--model-name', type=str,
                          help='Model filename')
    nds_group.add_argument('--device', type=str,
                          choices=['cpu', 'cuda'],
                          help='Device to use')
    nds_group.add_argument('--num-training', type=int,
                          help='Number of training instances')
    nds_group.add_argument('--num-validation', type=int,
                          help='Number of validation instances')
    nds_group.add_argument('--instance-size', type=int,
                          help='Number of customers per instance')
    nds_group.add_argument('--instance-file', type=str,
                          help='Path to VRP instance file')
    nds_group.add_argument('--max-iterations', type=int,
                          help='Maximum iterations for solving')
    nds_group.add_argument('--output-file', type=str,
                          help='Output file for solution')
    
    # VRPAgent-specific arguments
    vrpagent_group = parser.add_argument_group('VRPAgent Options')
    vrpagent_group.add_argument('--config', type=str,
                               help='Configuration file for VRPAgent')
    vrpagent_group.add_argument('--instances', type=str,
                               help='Path to instances directory')
    vrpagent_group.add_argument('--output-dir', type=str,
                               help='Output directory for results')
    
    # Catch-all for additional arguments
    parser.add_argument('extra_args', nargs=argparse.REMAINDER,
                       help='Additional arguments passed to the specific agent')
    
    args = parser.parse_args()
    
    # Route to appropriate agent
    if args.agent == 'vrpagent':
        return run_vrpagent(args)
    elif args.agent == 'nds':
        return run_nds(args)
    else:
        parser.error(f"Unknown agent: {args.agent}")


if __name__ == "__main__":
    sys.exit(main())

