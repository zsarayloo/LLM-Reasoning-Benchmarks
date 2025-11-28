import os
import sys
import json
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import re

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


class AdaptiveSteering:
    """
    Adaptive steering that dynamically adjusts α based on problem difficulty.
    """
    
    def __init__(self, layer_idx: int = 16, reasoning_vector_path: str = None):
        self.layer_idx = layer_idx
        
        # Load model
        print("[AdaptiveSteering] Loading model...")
        self.tokenizer, self.model = load_llama3_3b()
        self.device = self.model.device
        
        # Load reasoning vector
        if reasoning_vector_path is None:
            reasoning_vector_path = os.path.join(PROJECT_ROOT, "model", "reasoning_vector.npy")
        
        v_np = np.load(reasoning_vector_path)
        self.v = torch.from_numpy(v_np).float().to(self.device)
        self.v = self.v / (self.v.norm() + 1e-8)  # normalize
        
        # Layer-specific alpha ranges (from layer analysis)
        self.alpha_ranges = {
            8: (0.5, 1.5),
            10: (0.5, 1.5), 
            12: (0.3, 1.0),
            14: (0.2, 0.8),
            16: (0.1, 0.5)
        }
        
        print(f"[AdaptiveSteering] Loaded reasoning vector: {self.v.shape}")
        print(f"[AdaptiveSteering] Using layer {layer_idx} with α range {self.alpha_ranges.get(layer_idx, (0.1, 1.0))}")
    
    def assess_problem_difficulty(self, question: str) -> float:
        """
        Assess problem difficulty on a scale of 0.0 (easy) to 1.0 (hard).
        Uses heuristics based on problem characteristics.
        """
        difficulty_score = 0.0
        question_lower = question.lower()
        
        # Count mathematical operations
        operations = len(re.findall(r'[+\-×÷*/]', question))
        difficulty_score += min(operations * 0.15, 0.3)  # Max 0.3 for operations
        
        # Check for complex concepts
        complex_concepts = [
            'percentage', 'percent', '%', 'fraction', 'ratio', 'proportion',
            'area', 'perimeter', 'volume', 'speed', 'rate', 'average',
            'equation', 'solve', 'algebra', 'polynomial'
        ]
        concept_count = sum(1 for concept in complex_concepts if concept in question_lower)
        difficulty_score += min(concept_count * 0.2, 0.4)  # Max 0.4 for concepts
        
        # Check for large numbers
        numbers = re.findall(r'\d+', question)
        if numbers:
            max_number = max(int(num) for num in numbers)
            if max_number > 100:
                difficulty_score += 0.2
            elif max_number > 50:
                difficulty_score += 0.1
        
        # Check for multi-step problems (parentheses, multiple sentences)
        if '(' in question or ')' in question:
            difficulty_score += 0.15
        
        sentence_count = len([s for s in question.split('.') if s.strip()])
        if sentence_count > 2:
            difficulty_score += 0.1
        
        # Word problem complexity
        word_indicators = ['if', 'then', 'when', 'after', 'before', 'total', 'remaining', 'left']
        word_count = sum(1 for word in word_indicators if word in question_lower)
        difficulty_score += min(word_count * 0.05, 0.15)
        
        return min(difficulty_score, 1.0)  # Cap at 1.0
    
    def get_adaptive_alpha(self, question: str) -> float:
        """
        Get adaptive alpha value based on problem difficulty.
        Harder problems get smaller alpha (more conservative steering).
        """
        difficulty = self.assess_problem_difficulty(question)
        alpha_min, alpha_max = self.alpha_ranges.get(self.layer_idx, (0.1, 1.0))
        
        # Inverse relationship: harder problems get smaller alpha
        # difficulty=0.0 -> alpha_max, difficulty=1.0 -> alpha_min
        alpha = alpha_max - difficulty * (alpha_max - alpha_min)
        
        return alpha
    
    def build_cot_prompt(self, question: str) -> str:
        """Build CoT prompt for evaluation."""
        return (
            "You are a skilled mathematician. Solve this problem step by step.\n"
            "Show your reasoning clearly and double-check your work.\n"
            "At the end, provide your final numerical answer.\n\n"
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
    
    def generate_with_adaptive_steering(
        self, 
        question: str, 
        max_new_tokens: int = 256
    ) -> Tuple[str, str, float, float]:
        """
        Generate response with adaptive steering.
        Returns: (full_response, extracted_answer, difficulty, alpha_used)
        """
        # Assess difficulty and get adaptive alpha
        difficulty = self.assess_problem_difficulty(question)
        alpha = self.get_adaptive_alpha(question)
        
        prompt = self.build_cot_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Apply adaptive steering
        target_layer = self.model.model.layers[self.layer_idx]
        hook_handle = target_layer.register_forward_hook(self._make_steering_hook(alpha))
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        hook_handle.remove()
        
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        # Extract answer
        numbers = re.findall(r'-?\d+\.?\d*', response)
        extracted_answer = numbers[-1] if numbers else response.split()[-1] if response.split() else ""
        
        return response, extracted_answer, difficulty, alpha
    
    def compare_adaptive_vs_fixed_steering(
        self, 
        test_examples: List[Dict[str, Any]], 
        fixed_alpha: float = 0.3
    ) -> Dict[str, Any]:
        """
        Compare adaptive steering vs fixed alpha steering.
        """
        results = {
            "adaptive": {"correct": 0, "total": 0, "details": []},
            "fixed": {"correct": 0, "total": 0, "details": []},
            "baseline": {"correct": 0, "total": 0, "details": []}
        }
        
        print(f"[AdaptiveSteering] Comparing adaptive vs fixed (α={fixed_alpha}) vs baseline...")
        
        for i, example in enumerate(test_examples):
            question = example["question"]
            gold_answer = example["gold_answer"]
            
            try:
                # 1. Baseline (no steering)
                baseline_response, baseline_answer, _, _ = self.generate_with_adaptive_steering(question)
                baseline_correct = self._verify_answer(baseline_answer, gold_answer)
                
                # 2. Adaptive steering
                adaptive_response, adaptive_answer, difficulty, alpha_used = self.generate_with_adaptive_steering(question)
                adaptive_correct = self._verify_answer(adaptive_answer, gold_answer)
                
                # 3. Fixed alpha steering
                fixed_response, fixed_answer = self._generate_with_fixed_alpha(question, fixed_alpha)
                fixed_correct = self._verify_answer(fixed_answer, gold_answer)
                
                # Store results
                results["baseline"]["total"] += 1
                results["adaptive"]["total"] += 1
                results["fixed"]["total"] += 1
                
                if baseline_correct:
                    results["baseline"]["correct"] += 1
                if adaptive_correct:
                    results["adaptive"]["correct"] += 1
                if fixed_correct:
                    results["fixed"]["correct"] += 1
                
                # Store detailed results
                detail = {
                    "question": question,
                    "gold_answer": gold_answer,
                    "difficulty": difficulty,
                    "alpha_used": alpha_used,
                    "baseline_answer": baseline_answer,
                    "adaptive_answer": adaptive_answer,
                    "fixed_answer": fixed_answer,
                    "baseline_correct": baseline_correct,
                    "adaptive_correct": adaptive_correct,
                    "fixed_correct": fixed_correct
                }
                
                results["baseline"]["details"].append(detail)
                results["adaptive"]["details"].append(detail)
                results["fixed"]["details"].append(detail)
                
                if (i + 1) % 5 == 0:
                    print(f"[AdaptiveSteering] Processed {i+1}/{len(test_examples)} examples")
                
            except Exception as e:
                print(f"[AdaptiveSteering] Error processing example {i}: {e}")
                continue
        
        # Calculate accuracies
        for method in ["baseline", "adaptive", "fixed"]:
            total = results[method]["total"]
            correct = results[method]["correct"]
            results[method]["accuracy"] = correct / total if total > 0 else 0.0
        
        return results
    
    def _generate_with_fixed_alpha(self, question: str, alpha: float) -> Tuple[str, str]:
        """Generate with fixed alpha for comparison."""
        prompt = self.build_cot_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        target_layer = self.model.model.layers[self.layer_idx]
        hook_handle = target_layer.register_forward_hook(self._make_steering_hook(alpha))
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        hook_handle.remove()
        
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()
        
        numbers = re.findall(r'-?\d+\.?\d*', response)
        extracted_answer = numbers[-1] if numbers else response.split()[-1] if response.split() else ""
        
        return response, extracted_answer
    
    def _verify_answer(self, predicted: str, correct: str) -> bool:
        """Simple answer verification."""
        try:
            pred_num = float(predicted.strip())
            correct_num = float(correct.strip())
            return abs(pred_num - correct_num) < 0.01
        except:
            return predicted.strip().lower() == correct.strip().lower()
    
    def analyze_difficulty_distribution(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of problem difficulties."""
        difficulties = []
        alphas = []
        
        for example in examples:
            question = example["question"]
            difficulty = self.assess_problem_difficulty(question)
            alpha = self.get_adaptive_alpha(question)
            
            difficulties.append(difficulty)
            alphas.append(alpha)
        
        return {
            "difficulties": difficulties,
            "alphas": alphas,
            "difficulty_stats": {
                "mean": np.mean(difficulties),
                "std": np.std(difficulties),
                "min": np.min(difficulties),
                "max": np.max(difficulties)
            },
            "alpha_stats": {
                "mean": np.mean(alphas),
                "std": np.std(alphas),
                "min": np.min(alphas),
                "max": np.max(alphas)
            }
        }


def main():
    """Main adaptive steering evaluation."""
    # Load test examples
    test_metadata_path = os.path.join(PROJECT_ROOT, "test_metadata.json")
    if not os.path.exists(test_metadata_path):
        print(f"[adaptive_steering] Test metadata not found at {test_metadata_path}")
        return
    
    with open(test_metadata_path, "r") as f:
        test_examples = json.load(f)
    
    print(f"[adaptive_steering] Loaded {len(test_examples)} test examples")
    
    # Initialize adaptive steering
    adaptive_steerer = AdaptiveSteering(layer_idx=16)  # Use best layer from analysis
    
    # Analyze difficulty distribution
    print("\n=== DIFFICULTY ANALYSIS ===")
    difficulty_analysis = adaptive_steerer.analyze_difficulty_distribution(test_examples)
    
    print(f"Difficulty stats: {difficulty_analysis['difficulty_stats']}")
    print(f"Alpha stats: {difficulty_analysis['alpha_stats']}")
    
    # Compare adaptive vs fixed steering
    print("\n=== ADAPTIVE VS FIXED STEERING COMPARISON ===")
    comparison_results = adaptive_steerer.compare_adaptive_vs_fixed_steering(
        test_examples, fixed_alpha=0.3
    )
    
    # Print results
    print("\nResults Summary:")
    for method in ["baseline", "fixed", "adaptive"]:
        accuracy = comparison_results[method]["accuracy"]
        correct = comparison_results[method]["correct"]
        total = comparison_results[method]["total"]
        print(f"{method.capitalize():>10}: {accuracy:.3f} ({correct}/{total})")
    
    # Calculate improvements
    baseline_acc = comparison_results["baseline"]["accuracy"]
    fixed_acc = comparison_results["fixed"]["accuracy"]
    adaptive_acc = comparison_results["adaptive"]["accuracy"]
    
    print(f"\nImprovements over baseline:")
    print(f"Fixed steering:    {fixed_acc - baseline_acc:+.3f}")
    print(f"Adaptive steering: {adaptive_acc - baseline_acc:+.3f}")
    print(f"Adaptive vs Fixed: {adaptive_acc - fixed_acc:+.3f}")
    
    # Save results
    results_path = os.path.join(PROJECT_ROOT, "results", "adaptive_steering_results.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump({
            "difficulty_analysis": difficulty_analysis,
            "comparison_results": comparison_results
        }, f, indent=2)
    
    print(f"\n[adaptive_steering] Results saved to {results_path}")
    print("[adaptive_steering] Adaptive steering evaluation complete!")


if __name__ == "__main__":
    main()
