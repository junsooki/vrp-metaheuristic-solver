"""
Main script to run VRPAGENT
Discovers heuristics using LLM + Genetic Algorithm and solves VRP instances
"""

import os
import sys
from datetime import datetime
from importlib import import_module

# Add parent directory to path to import from common
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import modules
from vrpagent.f_generate_llm import LLMManager
from vrpagent.Heuristic import Heuristic
from common.data_loader import generate_random_instance
from common.problem import Solution

# Import GA module (has hyphen in filename)
GA_module = import_module('vrpagent.VRPAGENT-GA')


def main():
    """Main execution function"""
    
    print("="*70)
    print("VRPAGENT: LLM-Driven Heuristic Discovery for VRP")
    print("="*70)
    print()
    
    LLM_PROVIDER = 'groq'  
    LLM_MODEL = 'llama-3.3-70b-versatile'  

    
    # Get API key from environment
    if LLM_PROVIDER == 'groq':
        LLM_API_KEY = os.getenv('GROQ_API_KEY')
    elif LLM_PROVIDER == 'openai':
        LLM_API_KEY = os.getenv('OPENAI_API_KEY')
    else:  
        LLM_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    
    # GA settings - CHEAP DEFAULTS FOR TESTING
    N_INIT = 3   # Initial population size (was 10)
    N_E = 1      # Number of elites (was 3)
    N_C = 2      # Number of offspring per generation (was 7)
    MAX_GENERATIONS = 5  # Number of generations (was 20)
    
    # Problem settings
    NUM_TRAIN_INSTANCES = 3  # Training instances 
    NUM_TEST_INSTANCES = 2   # Test instances
    NUM_CUSTOMERS = 15       # Customers per instance 
    
    print(f"  LLM Provider: {LLM_PROVIDER}")
    print(f"  LLM Model: {LLM_MODEL}")
    print(f"  Initial Population: {N_INIT}")
    print(f"  Elites: {N_E}, Offspring: {N_C}")
    print(f"  Max Generations: {MAX_GENERATIONS}")
    print(f"  Training Instances: {NUM_TRAIN_INSTANCES}")
    print(f"  Customers per instance: {NUM_CUSTOMERS}")
    print()
    
    if not LLM_API_KEY:
        print("Error: No API key found!")
        if LLM_PROVIDER == 'groq':
            print("   Set GROQ_API_KEY environment variable")
            print("   Get free key at: https://console.groq.com/")
        elif LLM_PROVIDER == 'openai':
            print("   Set OPENAI_API_KEY environment variable")
        else:
            print("   Set ANTHROPIC_API_KEY environment variable")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # ========== Initialize LLM Manager ==========
    print("Initializing LLM Manager...")
    try:
        llm_manager = LLMManager(
            provider=LLM_PROVIDER,
            model=LLM_MODEL,
            api_key=LLM_API_KEY
        )
        print("LLM Manager initialized")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        return
    print()
    
    # ========== Generate Training Instances ==========
    print("Generating training instances...")
    train_instances = []
    for i in range(NUM_TRAIN_INSTANCES):
        instance = generate_random_instance(
            num_customers=NUM_CUSTOMERS,
            capacity=NUM_CUSTOMERS * 2,
            seed=i
        )
        train_instances.append(instance)
        print(f"  ✓ Instance {i+1}: {instance.name}")
    print()
    
    # ========== Set Global Variables for GA ==========
    GA_module.LLM = llm_manager
    GA_module.EvaluationInstances = train_instances
    GA_module.MaxGenerations = MAX_GENERATIONS
    GA_module.DefaultOrderingOperator = None  # Will use greedy insertion order
    
    # ========== Run Genetic Algorithm ==========
    print("Starting Genetic Algorithm to evolve heuristics...")
    print("="*70)
    print()
    
    try:
        # Generate initial population
        print("Step 1: Generating initial population...")
        population = GA_module.GenerateStartPop(N_INIT)
        
        if not population:
            print("Failed to generate initial population")
            return
        
        print(f"Generated {len(population)} initial heuristics")
        print()
        
        # Run GA
        print("Step 2: Running Genetic Algorithm...")
        best_heuristic = GA_module.GA(N_INIT, N_E, N_C)
        
        print()
        print("="*70)
        print("Genetic Algorithm Complete!")
        print(f"   Best Heuristic: {best_heuristic.name}")
        print(f"   Average Fitness: {best_heuristic.get_average_fitness():.4f}")
        print("="*70)
        print()
        
    except Exception as e:
        print(f"Error during GA: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========== Test Best Heuristic ==========
    print("Testing best heuristic on new instances...")
    print("-" * 70)
    
    test_instances = []
    for i in range(NUM_TEST_INSTANCES):
        instance = generate_random_instance(
            num_customers=NUM_CUSTOMERS,
            capacity=NUM_CUSTOMERS * 2,
            seed=1000 + i  # Different seed
        )
        test_instances.append(instance)
    
    test_results = []
    for i, instance in enumerate(test_instances):
        print(f"\nTest Instance {i+1}: {instance.name}")
        
        try:
            # Import LNS solver dynamically to avoid circular imports
            from importlib import import_module
            lns_module = import_module('vrpagent.VRPAgent-LNS')
            
            # Solve with best heuristic
            solution = lns_module.LNS_solver(
                I=instance,
                f_remove=best_heuristic.function,
                f_order=None,  # Will use default greedy
                max_iterations=1000
            )
            
            print(f"  Solution Cost: {solution.objective:.2f}")
            print(f"  Num Routes: {len([r for r in solution.routes if r])}")
            print(f"  Feasible: {solution.is_feasible()}")
            
            test_results.append(solution.objective)
            
        except Exception as e:
            print(f"Failed: {e}")
    
    print()
    print("="*70)
    print("Final Results:")
    print("-" * 70)
    if test_results:
        print(f"  Average Cost: {sum(test_results)/len(test_results):.2f}")
        print(f"  Best Cost: {min(test_results):.2f}")
        print(f"  Worst Cost: {max(test_results):.2f}")
    print("="*70)
    print()
    
    # ========== Save Best Heuristic ==========
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"best_heuristic_{timestamp}.py")
    
    with open(output_file, 'w') as f:
        f.write(f"# Best Heuristic: {best_heuristic.name}\n")
        f.write(f"# Fitness: {best_heuristic.get_average_fitness():.4f}\n")
        f.write(f"# Generated: {timestamp}\n\n")
        f.write(best_heuristic.code)
    
    print(f"Best heuristic saved to: {output_file}")
    print()
    print("✨ VRPAGENT execution complete!")


if __name__ == "__main__":
    main()

