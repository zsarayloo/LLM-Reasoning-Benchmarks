# LLM Reasoning Benchmarks

A comprehensive research project evaluating Large Language Model (LLM) reasoning capabilities across mathematical and optimization problems using various prompting strategies and activation steering techniques.

## 🎯 Overview

This repository implements and evaluates multiple reasoning strategies for LLMs on two primary benchmarks:
- **LiveMathBench**: Competition-style mathematical problems (AMC, CNMO, CCEE, WLPMC, hard problems)
- **NL4OPT**: Natural language optimization problems requiring linear programming solutions

The project explores various reasoning methodologies including Chain-of-Thought (CoT), Program-of-Thought (PoT), Self-Consistency, and novel activation steering approaches.

## 📊 Key Features

### Benchmarks
- **LiveMathBench**: 5 mathematical problem categories with step-by-step reasoning evaluation
- **NL4OPT**: 245 optimization problems with numerical verification using Gurobi solver

### Models Supported
- **GPT-5.1**: State-of-the-art reasoning via OpenAI API
- **Llama-3.2-3B**: Local inference with activation steering capabilities
- **Mistral-7B**: Alternative open-source model evaluation

### Reasoning Strategies
- **Chain-of-Thought (CoT)**: Step-by-step reasoning prompts
- **Program-of-Thought (PoT)**: Code generation for mathematical problem solving
- **Self-Consistency**: Multiple sampling with majority voting
- **Activation Steering**: Novel approach using reasoning vectors to guide model behavior

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Running Experiments

#### LiveMathBench Evaluation
```bash
# GPT-5 with Chain-of-Thought
python src/experiment/run_livemathbench_cot_gpt5.py

# Llama-3.2-3B with CoT
python src/experiment/run_livemathbench_cot_llama.py

# Self-Consistency approach
python src/experiment/run_livemathbench_cot_selfconsistency_gpt5.py
```

#### NL4OPT Evaluation
```bash
# Program-of-Thought approach
python src/experiment/run_nl4opt_pot_gpt5.py

# Linear Programming with Gurobi verification
python src/experiment/run_nl4opt_lp_gurobi_gpt5.py

# Enhanced strategies with self-checking
python src/experiment/run_nl4opt_lp_gurobi_selfcheck_gpt5.py
```

#### Activation Steering
```bash
# Collect reasoning data for vector construction
python src/experiment/collect_reasoning_data.py

# Build reasoning vector
python src/experiment/build_reasoning_vector.py

# Run steered inference
python src/experiment/steered_inference.py
```

## 📁 Project Structure

```
├── data/
│   ├── LiveMathBench/          # Mathematical competition problems
│   │   ├── AMC_en.jsonl        # American Mathematics Competitions
│   │   ├── CNMO_en.jsonl       # Chinese National Math Olympiad
│   │   ├── CCEE_en.jsonl       # Chinese College Entrance Exam
│   │   ├── WLPMC_en.jsonl      # William Lowell Putnam Competition
│   │   └── hard_en.jsonl       # Challenging problems
│   └── nl4opt/                 # Optimization problems dataset
├── src/
│   ├── experiment/             # Main experiment scripts
│   │   ├── run_*_gpt5.py      # GPT-5 evaluation scripts
│   │   ├── run_*_llama.py     # Llama model scripts
│   │   ├── steered_inference.py # Activation steering
│   │   ├── collect_reasoning_data.py # Data collection for steering
│   │   └── plot_*.py          # Visualization scripts
│   └── utils/                  # Utility functions
├── model/                      # Model loaders and reasoning vectors
├── results/                    # Experimental results and plots
└── figures/                    # Generated visualizations
```

## 🧠 Methodology

### Chain-of-Thought (CoT)
Prompts models to solve problems step-by-step with explicit reasoning chains:
```
"You are an expert competition mathematician.
Solve the following problem step by step.
At the very end, write: ANSWER: <final_answer>"
```

### Program-of-Thought (PoT)
Generates executable Python code to solve mathematical problems:
- Converts natural language problems into computational solutions
- Executes code to obtain numerical answers
- Particularly effective for optimization problems

### Self-Consistency
- Generates multiple reasoning paths for the same problem
- Uses majority voting or confidence-based selection
- Improves reliability on complex problems

### Activation Steering
Novel approach using internal model representations:
1. **Data Collection**: Extract hidden states from correct vs. incorrect reasoning
2. **Vector Construction**: Compute difference between positive and negative examples
3. **Steering**: Add scaled reasoning vector to model activations during inference

## 📈 Results Summary

### LiveMathBench Performance
- **GPT-5 CoT**: High accuracy across all problem categories
- **Self-Consistency**: Improved performance on challenging problems
- **Local Models**: Competitive performance with activation steering

### NL4OPT Performance
- **PoT Strategy**: ~81% accuracy on optimization problems
- **LP + Gurobi**: Strong performance with formal verification
- **Self-Check Methods**: Enhanced reliability through iterative refinement

### Key Findings
- Program-of-Thought excels on computational problems
- Self-consistency provides significant improvements on hard problems
- Activation steering enables smaller models to achieve competitive performance
- Formal verification (Gurobi) ensures solution correctness

## 🔧 Advanced Usage

### Custom Evaluation
```python
from src.experiment.data_loader_livemathbench import load_livemathbench_local
from src.experiment.nl4opt_utils import load_nl4opt

# Load specific benchmark splits
df_math = load_livemathbench_local(split="AMC", n_examples=50)
df_opt = load_nl4opt(n_examples=100)
```

### Activation Steering Parameters
```python
# Adjust steering strength
alpha = 3.0  # Positive values enhance reasoning
alpha = -3.0 # Negative values for ablation studies
```

## 📊 Visualization

Generate comprehensive result plots:
```bash
python src/experiment/plot_all_benchmarks.py
python src/experiment/plot_livemathbench_accuracy.py
python src/experiment/plot_results.py
```

## 🛠️ Dependencies

- **Core**: `torch`, `transformers`, `numpy`, `pandas`
- **APIs**: `openai` for GPT models
- **Optimization**: `gurobipy` for LP verification
- **Visualization**: `matplotlib`, `seaborn`
- **Utilities**: `python-dotenv`, `tqdm`, `scipy`, `sympy`

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{llm-reasoning-benchmarks,
  title={LLM Reasoning Benchmarks: Evaluating Mathematical and Optimization Problem Solving},
  author={Zahra Sarayloo},
  year={2025},
  url={https://github.com/zsarayloo/LLM-Reasoning-Benchmarks}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project requires significant computational resources for full evaluation. Consider using smaller sample sizes (`n_examples` parameter) for initial testing and development.
