# VRP Metaheuristic Solver

A comprehensive implementation of state-of-the-art metaheuristic approaches for Vehicle Routing Problems (VRPs), featuring both neural network-based and LLM-generated destruction heuristics.

## Overview

This repository implements two cutting-edge approaches to solving VRPs through learned Large Neighborhood Search (LNS) frameworks:

1. **Neural Deconstruction Search (NDS)** - Uses deep neural networks trained via reinforcement learning to learn which customers to remove from solutions
2. **VRPAGENT** - Uses Large Language Models (LLMs) to automatically generate and evolve destruction heuristics through genetic algorithms

Both approaches significantly outperform traditional handcrafted operations research methods while providing insights into automated heuristic design.


## References

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


## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original NDS implementation by Hottung et al.
- VRPAGENT framework by Hottung, Berto, Hua et al.
- Benchmark instances from Kool et al., Drakulic et al., and Ye et al.
- Computing resources provided by Paderborn Center for Parallel Computing (PC²)

## 📧 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/vrp-metaheuristic-solver/issues)
- Email: junsooki@usc.edu
- Discussions: [GitHub Discussions](https://github.com/yourusername/vrp-metaheuristic-solver/discussions)


