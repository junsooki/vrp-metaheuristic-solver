# VRP Metaheuristic Solver

A comprehensive implementation of state-of-the-art metaheuristic approaches for Vehicle Routing Problems (VRPs), featuring both neural network-based and LLM-generated destruction heuristics.

## 🚀 Overview

This repository implements two cutting-edge approaches to solving VRPs through learned Large Neighborhood Search (LNS) frameworks:

1. **Neural Deconstruction Search (NDS)** - Uses deep neural networks trained via reinforcement learning to learn which customers to remove from solutions
2. **VRPAGENT** - Uses Large Language Models (LLMs) to automatically generate and evolve destruction heuristics through genetic algorithms

Both approaches significantly outperform traditional handcrafted operations research methods while providing insights into automated heuristic design.

## 📊 Supported Problems

- **CVRP** - Capacitated Vehicle Routing Problem
- **VRPTW** - Vehicle Routing Problem with Time Windows
- **PCVRP** - Prize-Collecting Vehicle Routing Problem

## 🏗️ Architecture
```
vrp-metaheuristic-solver/
├── nds/                    # Neural Deconstruction Search
│   ├── models/            # Neural network architectures
│   ├── training/          # RL training pipeline
│   └── search/            # Augmented simulated annealing
├── vrpagent/              # LLM-based heuristic generation
│   ├── genetic/           # Genetic algorithm for evolution
│   ├── prompts/           # LLM prompts for code generation
│   └── operators/         # Generated C++ heuristics
├── common/                # Shared components
│   ├── lns/              # LNS framework
│   ├── instances/        # Problem instance generators
│   └── evaluation/       # Benchmarking utilities
├── benchmarks/            # Test instances and results
└── docs/                  # Documentation
```

## 🔬 Key Features

### Neural Deconstruction Search (NDS)
- **GPU-accelerated** parallel rollouts for fast solution generation
- **Transformer-based** architecture with solution encoding
- **Message passing** and tour encoding layers
- Processes **120k solutions/second** (vs 10k for traditional neural methods)
- Matches or exceeds state-of-the-art OR solvers (HGS, SISRs)

### VRPAGENT
- **CPU-only** execution at test time (no GPU required)
- **LLM-agnostic** framework (supports Gemini, GPT, Llama, Qwen, etc.)
- **Genetic algorithm** with elitism and biased crossover
- **Code length penalty** to prevent bloat and reduce LLM costs
- Generates **interpretable C++ code** that can be analyzed by experts

## 📈 Performance

| Method | CVRP-500 | CVRP-1000 | CVRP-2000 | Hardware |
|--------|----------|-----------|-----------|----------|
| HGS (baseline) | 36.66 | 41.51 | 57.38 | CPU |
| SISRs | 36.65 | 41.14 | 56.04 | CPU |
| NDS | **36.57** | **41.11** | **56.00** | CPU+GPU |
| VRPAGENT | **36.60** | **41.06** | **55.98** | CPU |

*Gap improvements of 0.2-0.3% represent significant savings in large-scale logistics*

## 🚀 Quick Start

### Prerequisites
```bash
# Python dependencies
pip install torch numpy pandas

# C++ compiler (for VRPAGENT)
# GCC 9+ or Clang 10+

# Optional: GPU support
# CUDA 11.0+ for NDS
```

### Installation
```bash
git clone https://github.com/yourusername/vrp-metaheuristic-solver.git
cd vrp-metaheuristic-solver
pip install -e .
```

### Running NDS
```python
from nds import NDSSolver

# Load or generate instance
instance = load_cvrp_instance("data/instances/cvrp500.json")

# Create solver
solver = NDSSolver(
    problem_type="cvrp",
    model_path="models/nds_cvrp500.pt"
)

# Solve
solution = solver.solve(instance, time_limit=60)
print(f"Solution cost: {solution.total_cost}")
```

### Running VRPAGENT
```python
from vrpagent import VRPAgent

# Initialize with LLM
agent = VRPAgent(
    llm_model="gemini-2.5-flash",
    problem_type="cvrp"
)

# Discovery phase (generates heuristics)
agent.discover(
    training_instances=training_set,
    num_iterations=40,
    population_size=100
)

# Solve with discovered heuristics
solution = agent.solve(instance, time_limit=60)
```

## 🧪 Training

### Training NDS Models
```bash
# Train for CVRP with 500 customers
python -m nds.train \
    --problem cvrp \
    --size 500 \
    --epochs 2000 \
    --gpu 0

# Training takes ~5-8 days on A100 GPU
```

### Discovering VRPAGENT Heuristics
```bash
# Run genetic algorithm to discover heuristics
python -m vrpagent.discover \
    --problem cvrp \
    --llm gemini-2.5-flash \
    --iterations 40 \
    --elite-size 10 \
    --offspring-size 30

# Discovery takes ~6-12 hours depending on LLM
# Costs: $0.50-$20 per run depending on model
```

## 📊 Benchmarking
```bash
# Run benchmarks comparing all methods
python scripts/benchmark.py \
    --problems cvrp vrptw pcvrp \
    --sizes 100 500 1000 2000 \
    --methods nds vrpagent hgs sisrs \
    --time-limit 60

# Results saved to benchmarks/results/
```

## 🔧 Advanced Usage

### Custom Problem Types
```python
# Extend for new VRP variant
from common.lns import BaseLNS

class MyCustomVRP(BaseLNS):
    def validate_solution(self, solution):
        # Custom constraint checking
        pass
    
    def greedy_insertion(self, customer, partial_solution):
        # Custom insertion logic
        pass
```

### Custom Destruction Heuristics
```cpp
// VRPAGENT: Write custom C++ operator
std::vector<int> select_by_custom(const Solution& sol) {
    // Your custom destruction logic
    std::vector<int> customers;
    // ... selection code ...
    return customers;
}
```

## 📚 References

### Neural Deconstruction Search
```bibtex
@article{hottung2025nds,
  title={Neural Deconstruction Search for Vehicle Routing Problems},
  author={Hottung, André and Wong-Chung, Paula and Tierney, Kevin},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```

### VRPAGENT
```bibtex
@article{hottung2025vrpagent,
  title={VRPAGENT: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems},
  author={Hottung, André and Berto, Federico and Hua, Chuanbo and others},
  journal={arXiv preprint arXiv:2510.07073},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution
- [ ] Additional VRP variants (VRPB, MDVRP, etc.)
- [ ] Support for more LLM providers
- [ ] Improved visualization tools
- [ ] Performance optimizations
- [ ] Additional baseline solvers

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Original NDS implementation by Hottung et al.
- VRPAGENT framework by Hottung, Berto, Hua et al.
- Benchmark instances from Kool et al., Drakulic et al., and Ye et al.
- Computing resources provided by Paderborn Center for Parallel Computing (PC²)

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/vrp-metaheuristic-solver/issues)
- Email: your.email@example.com
- Discussions: [GitHub Discussions](https://github.com/yourusername/vrp-metaheuristic-solver/discussions)

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Status**: 🚧 Under active development | **Version**: 0.1.0-alpha
```

---

## **Alternative Shorter Description** (if you want something punchier)
```
State-of-the-art VRP solvers using neural networks (NDS) and LLM-generated heuristics (VRPAGENT). Outperforms traditional OR methods on CVRP, VRPTW, and PCVRP.
