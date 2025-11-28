import os
import sys
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from experiment.analyze_livemathbench_cot_vs_sc import (
    normalize_answer_str as normalize_short_answer_str,
    verify_math_answer,
)

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
SRC_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.append(p)

from model.lama3_3b_loader import load_llama3_3b


class SteeringEvaluator:
    """
    Systematic evaluation of reasoning vector steering effects.
    """
    
    def __init__(self, layer_idx: int = 12, reasoning_vector_path: Optional[str] = None):
        self.layer_idx = layer_idx
        
        # Load model
        print("[SteeringEvaluator] Loading model...")
        self.tokenizer, self.model = load_llama3_3b()
        self.device = self.model.device
        
        # Load reasoning vector
        if reasoning_vector_path is None:
            reasoning_vector_path = os.path.join(PROJECT_ROOT, "model", "reasoning_vector.npy")
        
        if not os.path.exists(reasoning_vector_path):
            raise FileNotFoundError(f"Reasoning vector not found at {reasoning_vector_path}")
        
        v_np = np.load(reasoning_vector_path)
        self.v = torch.from_numpy(v_np).float().to(self.device)
        self.v = self.v / (self.v.norm() + 1e-8)  # normalize
        
        print(f"[SteeringEvaluator] Loaded reasoning vector: {self.v.shape}")
    
    def build_cot_prompt(self, question: str) -> str:
        """Build CoT prompt for evaluation."""
        return (
            "You are a careful mathematician. Think step by step and solve the problem.\n"
            "Show your work clearly and check your calculations.\n"
            "At the end, provide your final answer.\n\n"
            f"Question: {question}\n\n"
            "Solution:"
        )
    
    def _make_steering_hook(self, alpha: float):
        """Create steering hook for given alpha value."""
        def hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                hidden = output.clone()
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * self.v
                return hidden
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                hidden = output[0].clone()
                hidden[:, -1, :] = hidden[:, -1, :] + alpha * self.v
                return (hidden,) + output[1:]
            else:
                return output
        return hook
    
    def generate_with_steering(
        self, 
        question: str, 
        alpha: float = 0.0, 
        max_new_tokens: int = 256
    ) -> Tuple[str, str]:
        """
        Generate response with optional steering.
        Returns: (full_response, extracted_answer)
        """
        prompt = self.build_cot_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        hook_handle = None
        if alpha != 0.0:
            target_layer = self.model.model.layers[self.layer_idx]
            hook_handle = target_layer.register_forward_hook(self._make_steering_hook(alpha))
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        if hook_handle is not None:
            hook_handle.remove()
        
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        # Extract answer
        answer_line = response.splitlines()[-1] if response.splitlines() else ""
        if "Answer:" in answer_line:
            extracted_answer = answer_line.split("Answer:", 1)[1].strip()
        elif "=" in answer_line:
            parts = answer_line.split("=")
            extracted_answer = parts[-1].strip() if len(parts) > 1 else answer_line.strip()
        else:
            extracted_answer = answer_line.strip()
        
        return response, extracted_answer
    
    def evaluate_accuracy(
        self, 
        test_examples: List[Dict[str, Any]], 
        alpha_values: List[float],
        max_examples: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Evaluate accuracy across different steering strengths.
        """
        if max_examples is not None:
            test_examples = test_examples[:max_examples]
        
        results = {
            "alpha_values": alpha_values,
            "accuracies": [],
            "detailed_results": [],
            "examples": test_examples
        }
        
        print(f"[SteeringEvaluator] Evaluating {len(test_examples)} examples with {len(alpha_values)} alpha values...")
        
        for alpha in alpha_values:
            print(f"\n[SteeringEvaluator] Testing alpha = {alpha}")
            correct_count = 0
            alpha_results = []
            
            for i, example in enumerate(tqdm(test_examples, desc=f"Alpha {alpha}")):
                question = example["question"]
                gold_answer = example["gold_answer"]
                
                try:
                    # Generate response with steering
                    response, predicted_answer = self.generate_with_steering(question, alpha)
                    
                    # Evaluate correctness
                    pred_norm = normalize_short_answer_str(predicted_answer)
                    gold_norm = normalize_short_answer_str(gold_answer)
                    result = verify_math_answer(pred_norm, gold_norm)
                    is_correct = result["correct"]
                    
                    if is_correct:
                        correct_count += 1
                    
                    # Store detailed result
                    alpha_results.append({
                        "example_idx": i,
                        "question": question,
                        "gold_answer": gold_answer,
                        "predicted_answer": predicted_answer,
                        "response": response,
                        "is_correct": is_correct,
                        "alpha": alpha
                    })
                    
                except Exception as e:
                    print(f"[SteeringEvaluator] Error processing example {i} with alpha {alpha}: {e}")
                    alpha_results.append({
                        "example_idx": i,
                        "question": question,
                        "gold_answer": gold_answer,
                        "predicted_answer": "ERROR",
                        "response": f"Error: {e}",
                        "is_correct": False,
                        "alpha": alpha
                    })
            
            accuracy = correct_count / len(test_examples)
            results["accuracies"].append(accuracy)
            results["detailed_results"].extend(alpha_results)
            
            print(f"[SteeringEvaluator] Alpha {alpha}: Accuracy = {accuracy:.3f} ({correct_count}/{len(test_examples)})")
        
        return results
    
    def plot_accuracy_curve(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Plot accuracy vs steering strength."""
        alpha_values = results["alpha_values"]
        accuracies = results["accuracies"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(alpha_values, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Steering Strength (α)', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Reasoning Vector Steering: Accuracy vs Steering Strength', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Highlight baseline (alpha=0)
        baseline_idx = alpha_values.index(0.0) if 0.0 in alpha_values else None
        if baseline_idx is not None:
            plt.plot(alpha_values[baseline_idx], accuracies[baseline_idx], 'ro', 
                    markersize=12, label=f'Baseline (α=0): {accuracies[baseline_idx]:.3f}')
            plt.legend()
        
        # Add value annotations
        for i, (alpha, acc) in enumerate(zip(alpha_values, accuracies)):
            plt.annotate(f'{acc:.3f}', (alpha, acc), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(PROJECT_ROOT, "results", "steering_accuracy_curve.png")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SteeringEvaluator] Saved accuracy curve to {save_path}")
        plt.show()
    
    def analyze_by_category(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Analyze steering effects by problem category."""
        detailed_results = results["detailed_results"]
        
        # Simple category classification based on question content
        def classify_category(question: str) -> str:
            q_lower = question.lower()
            if any(word in q_lower for word in ["triangle", "circle", "angle", "area", "perimeter"]):
                return "geometry"
            elif any(word in q_lower for word in ["prime", "divisible", "remainder", "gcd", "lcm"]):
                return "number_theory"
            elif any(word in q_lower for word in ["equation", "solve", "polynomial", "factor"]):
                return "algebra"
            else:
                return "other"
        
        # Add categories to results
        for result in detailed_results:
            result["category"] = classify_category(result["question"])
        
        # Create summary DataFrame
        df = pd.DataFrame(detailed_results)
        summary = df.groupby(["alpha", "category"])["is_correct"].agg(["count", "sum", "mean"]).reset_index()
        summary.columns = ["alpha", "category", "total", "correct", "accuracy"]
        
        return summary
    
    def save_results(self, results: Dict[str, Any], save_path: str = None) -> None:
        """Save evaluation results to JSON."""
        if save_path is None:
            save_path = os.path.join(PROJECT_ROOT, "results", "steering_evaluation_results.json")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(save_path, "w") as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"[SteeringEvaluator] Saved results to {save_path}")


def main():
    """Main evaluation script."""
    # Configuration
    alpha_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]  # Range of steering strengths
    max_test_examples = 100  # Limit for faster evaluation
    
    # Load test metadata
    test_metadata_path = os.path.join(PROJECT_ROOT, "test_metadata.json")
    if not os.path.exists(test_metadata_path):
        print(f"[evaluate_steering_systematic] Test metadata not found at {test_metadata_path}")
        print("Please run collect_reasoning_data_enhanced.py first.")
        return
    
    with open(test_metadata_path, "r") as f:
        test_examples = json.load(f)
    
    print(f"[evaluate_steering_systematic] Loaded {len(test_examples)} test examples")
    
    # Initialize evaluator
    evaluator = SteeringEvaluator(layer_idx=12)
    
    # Run evaluation
    results = evaluator.evaluate_accuracy(
        test_examples=test_examples,
        alpha_values=alpha_values,
        max_examples=max_test_examples
    )
    
    # Analyze results
    print("\n=== STEERING EVALUATION RESULTS ===")
    for alpha, accuracy in zip(results["alpha_values"], results["accuracies"]):
        print(f"Alpha {alpha:5.1f}: Accuracy = {accuracy:.3f}")
    
    # Find best alpha
    best_idx = np.argmax(results["accuracies"])
    best_alpha = results["alpha_values"][best_idx]
    best_accuracy = results["accuracies"][best_idx]
    baseline_accuracy = results["accuracies"][results["alpha_values"].index(0.0)]
    
    print(f"\nBest steering: α = {best_alpha}, accuracy = {best_accuracy:.3f}")
    print(f"Baseline (α=0): accuracy = {baseline_accuracy:.3f}")
    print(f"Improvement: {best_accuracy - baseline_accuracy:.3f}")
    
    # Category analysis
    category_summary = evaluator.analyze_by_category(results)
    print("\n=== CATEGORY ANALYSIS ===")
    print(category_summary.to_string(index=False))
    
    # Save results
    evaluator.save_results(results)
    
    # Plot accuracy curve
    evaluator.plot_accuracy_curve(results)
    
    print("\n[evaluate_steering_systematic] Evaluation complete!")


if __name__ == "__main__":
    main()
