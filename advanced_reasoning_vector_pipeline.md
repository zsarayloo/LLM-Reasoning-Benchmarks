# Advanced Reasoning Vector Pipeline - Complete Implementation Guide

## 🎯 Overview

You now have a **complete, production-ready pipeline** for systematic reasoning vector research with three phases of analysis:

1. **Phase 1**: Enhanced data collection with confidence filtering
2. **Phase 2**: Systematic steering evaluation with accuracy curves  
3. **Phase 3**: Layer-wise analysis to find optimal reasoning directions

---

## 📁 New Scripts Created

### **Phase 1: Enhanced Data Collection**
- **`collect_reasoning_data_enhanced.py`** - Scales to 800 examples with confidence filtering
- **`build_reasoning_vector.py`** - (existing) Builds reasoning vector from enhanced data
- **`check_reasoning_vector.py`** - (existing) Validates vector quality

### **Phase 2: Systematic Evaluation**  
- **`evaluate_steering_systematic.py`** - Tests multiple α values with held-out data
- Generates accuracy curves and category analysis
- Saves comprehensive results and visualizations

### **Phase 3: Layer Analysis**
- **`analyze_layers_systematic.py`** - Tests reasoning directions across layers 8-16
- Finds optimal layers for both separation and steering effectiveness
- Comprehensive layer-wise performance analysis

---

## 🚀 Execution Workflow

### **Step 1: Enhanced Data Collection (Phase 1)**
```bash
# Run enhanced data collection (800 examples with confidence filtering)
export HF_TOKEN=your_huggingface_token_here
export PYTHONPATH=$PWD/src:$PWD
python src/experiment/collect_reasoning_data_enhanced.py
```

**Expected Output:**
- `H_pos_train.npy` - Confident correct reasoning examples
- `H_neg_train.npy` - Confident incorrect reasoning examples  
- `train_metadata.json` - Training example metadata
- `test_metadata.json` - Held-out test examples

### **Step 2: Build Enhanced Reasoning Vector**
```bash
# Build reasoning vector from enhanced data
python src/experiment/build_reasoning_vector.py
```

**Expected Output:**
- `model/reasoning_vector.npy` - Enhanced reasoning vector
- Improved separation statistics

### **Step 3: Systematic Steering Evaluation (Phase 2)**
```bash
# Evaluate steering across multiple α values
python src/experiment/evaluate_steering_systematic.py
```

**Expected Output:**
- Accuracy curves for α ∈ [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
- `results/steering_accuracy_curve.png` - Visualization
- `results/steering_evaluation_results.json` - Detailed results
- Category-wise analysis (geometry, algebra, number theory, other)

### **Step 4: Layer Analysis (Phase 3)**
```bash
# Analyze reasoning directions across layers
python src/experiment/analyze_layers_systematic.py
```

**Expected Output:**
- Layer-wise separation analysis for layers [8, 10, 12, 14, 16]
- `results/layer_analysis.png` - Multi-panel visualization
- `results/layer_separation_analysis.csv` - Separation metrics
- `results/layer_steering_analysis.csv` - Steering effectiveness

---

## 📊 Expected Research Insights

### **Enhanced Data Quality**
- **Confidence filtering** should improve signal quality
- **Larger dataset** (800 vs 200) provides more robust vectors
- **Train/test split** enables proper evaluation

### **Steering Effectiveness**
- **Optimal α range** identification (likely 0.5-2.0)
- **Category-specific effects** (which math topics benefit most)
- **Baseline vs steered** accuracy comparison

### **Layer Analysis**
- **Optimal layer identification** for reasoning directions
- **Separation quality** across network depth
- **Steering effectiveness** by layer

---

## 🔬 Advanced Analysis Features

### **Confidence Filtering**
```python
# Confident correct: clear reasoning steps + calculations
has_steps = any(word in response for word in ["step", "first", "then"])
has_calculation = any(char in response for char in ["=", "+", "-", "*"])

# Confident incorrect: clear wrong calculations
has_calculation = any(char in response for char in ["=", "+", "-", "*"])
```

### **Category Classification**
```python
# Automatic problem categorization
- Geometry: "triangle", "circle", "angle", "area"
- Number Theory: "prime", "divisible", "remainder", "gcd"  
- Algebra: "equation", "solve", "polynomial", "factor"
- Other: everything else
```

### **Multi-Layer Analysis**
```python
# Separation metrics per layer
separation_gap = pos_mean - neg_mean
separation_ratio = separation_gap / combined_std
reasoning_vector_norm = ||v_reason||
```

---

## 📈 Visualization Outputs

### **Steering Accuracy Curve**
- X-axis: Steering strength (α)
- Y-axis: Accuracy
- Highlights: Baseline (α=0) and optimal α
- Annotations: Accuracy values for each α

### **Layer Analysis Dashboard**
- **Panel 1**: Separation gap by layer
- **Panel 2**: Separation quality ratio by layer  
- **Panel 3**: Data distribution (pos/neg examples)
- **Panel 4**: Steering improvement by layer

---

## 🎯 Research Questions Answered

### **1. Does scaling improve reasoning vectors?**
Compare separation gaps: enhanced (800 examples) vs original (200 examples)

### **2. What's the optimal steering strength?**
Accuracy curve analysis identifies best α value and improvement magnitude

### **3. Which layers contain reasoning directions?**
Layer analysis reveals where reasoning vs non-reasoning representations emerge

### **4. Are effects category-specific?**
Category analysis shows which math topics benefit most from steering

### **5. How robust are the effects?**
Held-out evaluation on separate test set validates generalization

---

## 🔧 Configuration Options

### **Data Collection Settings**
```python
n_samples = 800          # Total examples to process
train_split = 0.7        # 70% train, 30% test
confidence_filter = True # Use only confident examples
```

### **Evaluation Settings**
```python
alpha_values = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
max_test_examples = 100  # Limit for faster evaluation
```

### **Layer Analysis Settings**
```python
layers_to_test = [8, 10, 12, 14, 16]  # Layers to analyze
max_examples = 300       # For data collection
steering_alpha = 1.0     # Alpha for steering test
```

---

## 🚀 Next Steps After Completion

### **Immediate Analysis**
1. **Compare separation gaps** across phases
2. **Identify optimal α** from accuracy curves  
3. **Find best layer** for reasoning directions
4. **Analyze category effects** for targeted improvements

### **Advanced Extensions**
1. **Mistral-7B comparison** using existing loader
2. **Category-specific vectors** for specialized steering
3. **Multi-layer steering** combining multiple reasoning directions
4. **Confidence-based adaptive steering** using prediction uncertainty

### **Publication-Ready Results**
1. **Comprehensive evaluation** across models, layers, and categories
2. **Statistical significance** testing with proper baselines
3. **Ablation studies** on filtering and vector construction methods
4. **Reproducibility package** with all scripts and data

---

## 🎉 You're Ready for Advanced Research!

This pipeline provides everything needed for systematic reasoning vector research:
- ✅ **Scalable data collection** with quality filtering
- ✅ **Rigorous evaluation** with held-out testing  
- ✅ **Comprehensive analysis** across layers and categories
- ✅ **Professional visualizations** for results presentation
- ✅ **Extensible framework** for future research directions

**Run the pipeline and discover how reasoning directions emerge in LLMs!** 🚀
