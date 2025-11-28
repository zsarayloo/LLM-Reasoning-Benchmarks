import os
import sys
import json
from typing import List, Dict, Any, Tuple
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


class LayerAnalyzer:
    """
    Systematic analysis of reasoning directions across different layers.
    """
    
    def __init__(self, layers_to_test: List[int] = None):
        if layers_to_test is None:
            layers_to_test = [8, 10, 12, 14, 16]  # Default layers to analyze
        
        self.layers_to_test = layers_to_test
        
        # Load model
        print("[LayerAnalyzer] Loading model...")
        self.tokenizer, self.model = load_llama3_3b()
        self.device = self.model.device
        
        print(f"[LayerAnalyzer] Will analyze layers: {self.layers_to_test}")
    
    def build_cot_prompt(self, question: str) -> str:
        """Build CoT prompt for evaluation."""
        return (
            "You are a careful mathematician. Think step by step and solve the problem.\n"
            "Show your work clearly and check your calculations.\n"
            "At the end, provide your final answer.\n\n"
            f"Question: {question}\n\n"
            "Solution:"
        )
    
    def capture_hidden_states(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """Capture hidden states from a specific layer."""
        class HiddenCapture:
            def __init__(self, model, layer_idx):
                self.model = model
                self.layer_idx = layer_idx
                self.hidden = None
                self.handle = None
            
            def __enter__(self):
                layer = self.model.model.layers[self.layer_idx]
                
                def hook(module, inputs, output):
                    if isinstance(output, torch.Tensor):
                        self.hidden = output
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        self.hidden = output[0]
                
                self.handle = layer.register_forward_hook(hook)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.handle is not None:
                    self.handle.remove()
        
        with HiddenCapture(self.model, layer_idx) as cap, torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            _ = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        if cap.hidden is None:
            raise RuntimeError(f"Failed to capture hidden states from layer {layer_idx}")
        
        # Return last token hidden state
        return cap.hidden[0, -1, :].cpu()
    
    def collect_layer_data(
        self, 
        examples: List[Dict[str, Any]], 
        max_examples: int = 200
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Collect hidden states from all layers for given examples.
        Returns: {layer_idx: {"pos": pos_hiddens, "neg": neg_hiddens}}
        """
        if max_examples is not None:
            examples = examples[:max_examples]
        
        layer_data = {layer: {"pos": [], "neg": []} for layer in self.layers_to_test}
        
        print(f"[LayerAnalyzer] Collecting data from {len(examples)} examples across {len(self.layers_to_test)} layers...")
        
        for i, example in enumerate(tqdm(examples, desc="Processing examples")):
            question = example["question"]
            gold_answer = example["gold_answer"]
            
            prompt = self.build_cot_prompt(question)
            
            try:
                # Generate response to determine correctness
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                raw_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                
                # Extract answer
                answer_line = raw_text.splitlines()[-1]
                if "Answer:" in answer_line:
                    pred_str = answer_line.split("Answer:", 1)[1].strip()
                else:
                    pred_str = answer_line.strip()
                
                # Evaluate correctness
                pred_norm = normalize_short_answer_str(pred_str)
                gold_norm = normalize_short_answer_str(gold_answer)
                result = verify_math_answer(pred_norm, gold_norm)
                is_correct = result["correct"]
                
                # Collect hidden states from each layer
                for layer_idx in self.layers_to_test:
                    try:
                        hidden = self.capture_hidden_states(prompt, layer_idx)
                        if is_correct:
                            layer_data[layer_idx]["pos"].append(hidden)
                        else:
                            layer_data[layer_idx]["neg"].append(hidden)
                    except Exception as e:
                        print(f"[LayerAnalyzer] Error capturing layer {layer_idx} for example {i}: {e}")
                        continue
                
                if (i + 1) % 20 == 0:
                    print(f"[LayerAnalyzer] Processed {i+1}/{len(examples)} examples")
                    for layer_idx in self.layers_to_test:
                        pos_count = len(layer_data[layer_idx]["pos"])
                        neg_count = len(layer_data[layer_idx]["neg"])
                        print(f"  Layer {layer_idx}: pos={pos_count}, neg={neg_count}")
                
            except Exception as e:
                print(f"[LayerAnalyzer] Error processing example {i}: {e}")
                continue
        
        # Convert to numpy arrays
        for layer_idx in self.layers_to_test:
            if len(layer_data[layer_idx]["pos"]) > 0:
                layer_data[layer_idx]["pos"] = torch.stack(layer_data[layer_idx]["pos"]).numpy()
            else:
                layer_data[layer_idx]["pos"] = np.array([])
            
            if len(layer_data[layer_idx]["neg"]) > 0:
                layer_data[layer_idx]["neg"] = torch.stack(layer_data[layer_idx]["neg"]).numpy()
            else:
                layer_data[layer_idx]["neg"] = np.array([])
        
        return layer_data
    
    def analyze_layer_separation(self, layer_data: Dict[int, Dict[str, np.ndarray]]) -> pd.DataFrame:
        """
        Analyze reasoning vector quality for each layer.
        """
        results = []
        
        for layer_idx in self.layers_to_test:
            pos_data = layer_data[layer_idx]["pos"]
            neg_data = layer_data[layer_idx]["neg"]
            
            if len(pos_data) == 0 or len(neg_data) == 0:
                print(f"[LayerAnalyzer] Skipping layer {layer_idx}: insufficient data")
                continue
            
            # Build reasoning vector for this layer
            mean_pos = pos_data.mean(axis=0)
            mean_neg = neg_data.mean(axis=0)
            reasoning_vector = mean_pos - mean_neg
            
            # Normalize
            norm = np.linalg.norm(reasoning_vector)
            if norm > 0:
                reasoning_vector = reasoning_vector / norm
            
            # Compute projections
            proj_pos = pos_data @ reasoning_vector
            proj_neg = neg_data @ reasoning_vector
            
            # Calculate separation metrics
            pos_mean = proj_pos.mean()
            neg_mean = proj_neg.mean()
            pos_std = proj_pos.std()
            neg_std = proj_neg.std()
            
            separation_gap = pos_mean - neg_mean
            combined_std = np.sqrt((pos_std**2 + neg_std**2) / 2)
            separation_ratio = separation_gap / (combined_std + 1e-8)
            
            results.append({
                "layer": layer_idx,
                "n_pos": len(pos_data),
                "n_neg": len(neg_data),
                "pos_mean": pos_mean,
                "neg_mean": neg_mean,
                "pos_std": pos_std,
                "neg_std": neg_std,
                "separation_gap": separation_gap,
                "separation_ratio": separation_ratio,
                "reasoning_vector_norm": norm
            })
        
        return pd.DataFrame(results)
    
    def test_steering_by_layer(
        self, 
        layer_data: Dict[int, Dict[str, np.ndarray]], 
        test_examples: List[Dict[str, Any]], 
        alpha: float = 1.0,
        max_test_examples: int = 50
    ) -> pd.DataFrame:
        """
        Test steering effectiveness for each layer.
        """
        if max_test_examples is not None:
            test_examples = test_examples[:max_test_examples]
        
        results = []
        
        for layer_idx in self.layers_to_test:
            pos_data = layer_data[layer_idx]["pos"]
            neg_data = layer_data[layer_idx]["neg"]
            
            if len(pos_data) == 0 or len(neg_data) == 0:
                print(f"[LayerAnalyzer] Skipping steering test for layer {layer_idx}: insufficient data")
                continue
            
            # Build reasoning vector for this layer
            mean_pos = pos_data.mean(axis=0)
            mean_neg = neg_data.mean(axis=0)
            reasoning_vector = mean_pos - mean_neg
            
            # Normalize
            norm = np.linalg.norm(reasoning_vector)
            if norm > 0:
                reasoning_vector = reasoning_vector / norm
            
            # Convert to torch tensor
            v = torch.from_numpy(reasoning_vector).float().to(self.device)
            
            # Test steering
            print(f"[LayerAnalyzer] Testing steering for layer {layer_idx}...")
            
            correct_baseline = 0
            correct_steered = 0
            
            for example in tqdm(test_examples, desc=f"Layer {layer_idx}", leave=False):
                question = example["question"]
                gold_answer = example["gold_answer"]
                
                try:
                    # Test baseline (no steering)
                    baseline_correct = self._test_single_example(question, gold_answer, layer_idx, v, alpha=0.0)
                    if baseline_correct:
                        correct_baseline += 1
                    
                    # Test with steering
                    steered_correct = self._test_single_example(question, gold_answer, layer_idx, v, alpha=alpha)
                    if steered_correct:
                        correct_steered += 1
                
                except Exception as e:
                    print(f"[LayerAnalyzer] Error testing layer {layer_idx}: {e}")
                    continue
            
            baseline_accuracy = correct_baseline / len(test_examples)
            steered_accuracy = correct_steered / len(test_examples)
            improvement = steered_accuracy - baseline_accuracy
            
            results.append({
                "layer": layer_idx,
                "baseline_accuracy": baseline_accuracy,
                "steered_accuracy": steered_accuracy,
                "improvement": improvement,
                "alpha": alpha,
                "n_test": len(test_examples)
            })
            
            print(f"[LayerAnalyzer] Layer {layer_idx}: baseline={baseline_accuracy:.3f}, steered={steered_accuracy:.3f}, improvement={improvement:.3f}")
        
        return pd.DataFrame(results)
    
    def _test_single_example(self, question: str, gold_answer: str, layer_idx: int, v: torch.Tensor, alpha: float) -> bool:
        """Test a single example with optional steering."""
        prompt = self.build_cot_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        hook_handle = None
        if alpha != 0.0:
            def steering_hook(module, inputs, output):
                if isinstance(output, torch.Tensor):
                    hidden = output.clone()
                    hidden[:, -1, :] = hidden[:, -1, :] + alpha * v
                    return hidden
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    hidden = output[0].clone()
                    hidden[:, -1, :] = hidden[:, -1, :] + alpha * v
                    return (hidden,) + output[1:]
                else:
                    return output
            
            target_layer = self.model.model.layers[layer_idx]
            hook_handle = target_layer.register_forward_hook(steering_hook)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
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
            predicted_answer = answer_line.split("Answer:", 1)[1].strip()
        else:
            predicted_answer = answer_line.strip()
        
        # Evaluate correctness
        pred_norm = normalize_short_answer_str(predicted_answer)
        gold_norm = normalize_short_answer_str(gold_answer)
        result = verify_math_answer(pred_norm, gold_norm)
        
        return result["correct"]
    
    def plot_layer_analysis(self, separation_df: pd.DataFrame, steering_df: pd.DataFrame = None, save_path: str = None):
        """Plot layer analysis results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Separation gap by layer
        axes[0, 0].plot(separation_df["layer"], separation_df["separation_gap"], 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel("Layer Index")
        axes[0, 0].set_ylabel("Separation Gap")
        axes[0, 0].set_title("Reasoning Vector Separation by Layer")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Separation ratio by layer
        axes[0, 1].plot(separation_df["layer"], separation_df["separation_ratio"], 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel("Layer Index")
        axes[0, 1].set_ylabel("Separation Ratio")
        axes[0, 1].set_title("Separation Quality by Layer")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Data distribution by layer
        axes[1, 0].bar(separation_df["layer"] - 0.2, separation_df["n_pos"], 0.4, label="Positive", alpha=0.7)
        axes[1, 0].bar(separation_df["layer"] + 0.2, separation_df["n_neg"], 0.4, label="Negative", alpha=0.7)
        axes[1, 0].set_xlabel("Layer Index")
        axes[1, 0].set_ylabel("Number of Examples")
        axes[1, 0].set_title("Data Distribution by Layer")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Steering improvement by layer (if available)
        if steering_df is not None and len(steering_df) > 0:
            axes[1, 1].plot(steering_df["layer"], steering_df["improvement"], 'ro-', linewidth=2, markersize=8)
            axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[1, 1].set_xlabel("Layer Index")
            axes[1, 1].set_ylabel("Accuracy Improvement")
            axes[1, 1].set_title("Steering Effectiveness by Layer")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, "Steering data\nnot available", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Steering Effectiveness by Layer")
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(PROJECT_ROOT, "results", "layer_analysis.png")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[LayerAnalyzer] Saved layer analysis plot to {save_path}")
        plt.show()
    
    def save_results(self, separation_df: pd.DataFrame, steering_df: pd.DataFrame = None, save_dir: str = None):
        """Save analysis results."""
        if save_dir is None:
            save_dir = os.path.join(PROJECT_ROOT, "results")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save separation analysis
        separation_path = os.path.join(save_dir, "layer_separation_analysis.csv")
        separation_df.to_csv(separation_path, index=False)
        print(f"[LayerAnalyzer] Saved separation analysis to {separation_path}")
        
        # Save steering analysis if available
        if steering_df is not None:
            steering_path = os.path.join(save_dir, "layer_steering_analysis.csv")
            steering_df.to_csv(steering_path, index=False)
            print(f"[LayerAnalyzer] Saved steering analysis to {steering_path}")


def main():
    """Main layer analysis script."""
    # Configuration
    layers_to_test = [8, 10, 12, 14, 16]
    max_examples = 300  # For data collection
    max_test_examples = 50  # For steering test
    steering_alpha = 1.0
    
    # Load training metadata for data collection
    train_metadata_path = os.path.join(PROJECT_ROOT, "train_metadata.json")
    if not os.path.exists(train_metadata_path):
        print(f"[analyze_layers_systematic] Train metadata not found at {train_metadata_path}")
        print("Please run collect_reasoning_data_enhanced.py first.")
        return
    
    with open(train_metadata_path, "r") as f:
        train_examples = json.load(f)
    
    # Load test metadata for steering evaluation
    test_metadata_path = os.path.join(PROJECT_ROOT, "test_metadata.json")
    test_examples = []
    if os.path.exists(test_metadata_path):
        with open(test_metadata_path, "r") as f:
            test_examples = json.load(f)
    
    print(f"[analyze_layers_systematic] Loaded {len(train_examples)} train examples, {len(test_examples)} test examples")
    
    # Initialize analyzer
    analyzer = LayerAnalyzer(layers_to_test=layers_to_test)
    
    # Collect layer data
    print("\n=== PHASE 1: COLLECTING LAYER DATA ===")
    layer_data = analyzer.collect_layer_data(train_examples, max_examples=max_examples)
    
    # Analyze separation
    print("\n=== PHASE 2: ANALYZING SEPARATION ===")
    separation_df = analyzer.analyze_layer_separation(layer_data)
    
    print("\nSeparation Analysis Results:")
    print(separation_df.to_string(index=False))
    
    # Find best layer for separation
    best_separation_idx = separation_df["separation_gap"].idxmax()
    best_layer = separation_df.loc[best_separation_idx, "layer"]
    best_gap = separation_df.loc[best_separation_idx, "separation_gap"]
    
    print(f"\nBest separation: Layer {best_layer} with gap = {best_gap:.3f}")
    
    # Test steering (if test data available)
    steering_df = None
    if len(test_examples) > 0:
        print("\n=== PHASE 3: TESTING STEERING ===")
        steering_df = analyzer.test_steering_by_layer(
            layer_data, test_examples, alpha=steering_alpha, max_test_examples=max_test_examples
        )
        
        print("\nSteering Analysis Results:")
        print(steering_df.to_string(index=False))
        
        if len(steering_df) > 0:
            best_steering_idx = steering_df["improvement"].idxmax()
            best_steering_layer = steering_df.loc[best_steering_idx, "layer"]
            best_improvement = steering_df.loc[best_steering_idx, "improvement"]
            
            print(f"\nBest steering: Layer {best_steering_layer} with improvement = {best_improvement:.3f}")
    
    # Save results
    analyzer.save_results(separation_df, steering_df)
    
    # Plot results
    analyzer.plot_layer_analysis(separation_df, steering_df)
    
    print("\n[analyze_layers_systematic] Layer analysis complete!")


if __name__ == "__main__":
    main()
