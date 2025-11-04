# Comprehensive LLM Statistical Reasoning Evaluation Platform

A sophisticated research platform for evaluating and comparing Large Language Models' (LLMs) statistical reasoning capabilities through interactive hypothesis testing with multiple prompting strategies.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![W&B](https://img.shields.io/badge/Weights_&_Biases-Experiment_Tracking-orange)](https://wandb.ai/)

## 🎯 Overview

This platform provides a comprehensive framework for:
- **Interactive Statistical Test Configuration**: Hierarchical menu system for complete test parameter setup
- **LLM Performance Evaluation**: Compare OpenAI GPT-4 and DeepSeek models on statistical reasoning tasks
- **Multiple Prompting Strategies**: Test four research-based prompt variations (Standard, Chain-of-Thought, Expert, Template)
- **Ground Truth Validation**: SciPy-based statistical calculations for accuracy assessment
- **Experiment Tracking**: Automated logging with Weights & Biases (W&B)
- **Flexible Data Generation**: Support for multiple distributions with customizable constraints

## ✨ Key Features

### 🔧 **Interactive Configuration System**
- **Sample Comparison Types**: One-sample vs. Two-sample tests
- **Test Tail Selection**: Two-tailed, left-tailed, or right-tailed tests
- **Hypothesis Management**: Auto-generation or custom hypothesis input
- **Distribution Support**: Normal, Uniform, Binomial, and Poisson distributions
- **Data Constraints**: Optional min/max value constraints during generation
- **Custom Prompts**: User-defined prompts with automatic structure enforcement

### 📊 **Statistical Analysis Capabilities**
- **One-Sample t-tests**: Compare sample mean to known population mean
- **Two-Sample t-tests**: Compare means of two independent samples
- **Levene's Test**: Automatic variance equality assessment for two-sample tests
- **Comprehensive Metrics**: Mean, standard deviation, t-statistic, p-value, critical values, confidence intervals
- **Multiple Distributions**: Support for Normal, Uniform, Binomial, and Poisson data generation

### 🤖 **LLM Integration**
- **OpenAI GPT-4**: Latest GPT-4 model with structured prompting
- **DeepSeek Chat**: DeepSeek's latest reasoning model
- **Four Prompt Styles**:
  1. **Standard**: Clear 4-section structured format
  2. **Chain-of-Thought**: Step-by-step explicit reasoning
  3. **Expert Role-Playing**: Senior biostatistician perspective
  4. **Template**: Visual formatting with error prevention
- **Automatic JSON Parsing**: Extract p-values and statistical decisions from LLM responses

### 📈 **Enhanced Analytics**
- **Ground Truth Comparison**: Side-by-side comparison with SciPy calculations
- **Accuracy Assessment**: Automatic verification of LLM conclusions
- **Detailed Response Display**: Full structured analysis from each model
- **Performance Metrics**: Response quality, reasoning transparency, calculation accuracy

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
```

### Installation

1. **Clone the repository** (or download the files)
```bash
git clone https://github.com/YOUR_USERNAME/llm-statistical-reasoning.git
cd llm-statistical-reasoning
```

2. **Create virtual environment** (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Dependencies

```
numpy>=1.24.0
scipy>=1.10.0
requests>=2.31.0
openai>=1.0.0
wandb>=0.16.0
```

### API Keys Setup

Set your API keys as environment variables:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:DEEPSEEK_API_KEY = "your-deepseek-api-key"
```

**Windows (Command Prompt):**
```cmd
set OPENAI_API_KEY=your-openai-api-key
set DEEPSEEK_API_KEY=your-deepseek-api-key
```

**macOS/Linux:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
```

**Permanent Setup (Optional):**
Create a `.env` file in the project directory:
```
OPENAI_API_KEY=your-openai-api-key
DEEPSEEK_API_KEY=your-deepseek-api-key
```

### Running the Demo

```bash
python demo_2.0.py
```

## 📖 Usage Guide

### Interactive Configuration Workflow

The platform guides you through a comprehensive 7-step configuration process:

#### **Step 1: Sample Comparison Type**
```
1. SAMPLE COMPARISON TYPE:
   1. One-sample test (compare sample mean to known value)
   2. Two-sample test (compare two sample means)
Select comparison type (1-2, default=1):
```

#### **Step 2: Test Tail Type**
```
2. TEST TAIL TYPE:
   1. Two-tailed test (≠, checks for difference in either direction)
   2. Left-tailed test (<, checks if sample is significantly smaller)
   3. Right-tailed test (>, checks if sample is significantly larger)
Select tail type (1-3, default=1):
```

#### **Step 3: Hypothesis Configuration**
```
3. HYPOTHESIS CONFIGURATION:
Enter null hypothesis (default: μ = 10.0):
   Null hypothesis (H₀): μ = 10.0
   Auto-generated alternate: μ ≠ 10.0
Use auto-generated alternate hypothesis? (y/n, default=y):
```

#### **Step 4: Prompt Selection**
```
4. PROMPT SELECTION:
Use default prompt template? (y/n, default=y): n

   Prompt Style Options:
   1. Standard (clear 4-section structure)
   2. Chain-of-Thought (step-by-step reasoning)
   3. Expert Role-Playing (senior biostatistician perspective)
   4. Structured Template (visual formatting with error prevention)
Select prompt style (1-4, default=1):
```

#### **Step 5: Data Generation Settings**
```
5. DATA GENERATION SETTINGS:
   Distribution types:
   1. Normal distribution
   2. Uniform distribution
   3. Binomial distribution
   4. Poisson distribution
Select distribution (1-4, default=1):

   Parameters for NORMAL distribution:
   Mean (default=10.0):
   Standard deviation (default=5.0):

   Data constraints (optional):
   Minimum value constraint (press Enter for none):
   Maximum value constraint (press Enter for none):
```

#### **Step 6: Sample Size Specification**
```
6. SAMPLE SIZE SPECIFICATION:
Sample size for first sample (default=30):
Random seed (default=42):
```

#### **Step 7: Configuration Review**
```
7. CONFIGURATION REVIEW:
   ==================================================
   Test Type: One-Sample two-tailed
   Null Hypothesis: μ = 10.0
   Alternate Hypothesis: μ ≠ 10.0
   Distribution: Normal
   Parameters: {'mean': 10.0, 'std': 5.0}
   Sample Size(s): 30
   Prompt: Chain-of-Thought
   Random Seed: 42
   ==================================================

Proceed with this configuration? (y/n, default=y):
```

### Example Output

```
============================================================
GROUND TRUTH ANALYSIS (One-Sample Test):
============================================================
Sample Statistics:
  Sample mean: 9.0593
  Sample std dev: 4.5000
  Sample size: 30
  Degrees of freedom: 29

Hypothesis Test:
  Null hypothesis: μ = 10.0000
  Significance level: α = 0.050
  T-statistic: -1.1450
  P-value: 0.261564 (two-sided)

Critical Values (α = 0.050):
  Critical values: ±2.045
  Lower critical: -2.045
  Upper critical: 2.045

Conclusion: Fail to reject H₀ at α = 0.050

============================================================
TESTING LLM RESPONSES
============================================================

==================================================
OPENAI GPT-4 FULL RESPONSE:
==================================================
1. TEST SELECTION: 
   - The appropriate statistical test is the one-sample t-test
   - This should be a two-tailed test (testing for any difference)
   - Alternative hypothesis (H₁): μ ≠ 10.0

2. CALCULATIONS:
   - Sample mean (x̄) = 9.06
   - Standard deviation (s) = 4.50
   - Sample size (n) = 30
   - Test statistic: t = (9.06 - 10.0) / (4.50 / √30) = -1.14
   - P-value = 0.26

3. RESULTS:
   - Test statistic value: -1.14
   - P-value: 0.26
   - Critical values: ±2.045
   - Confidence interval: (7.38, 10.74)

4. CONCLUSION:
   Since p-value (0.26) > α (0.05), we fail to reject H₀

{"p_value": 0.26, "conclusion": "fail to reject", "test_type": "two-tailed", "alternative_hypothesis": "μ ≠ 10.0"}

==================================================
DEEPSEEK FULL RESPONSE:
==================================================
[Similar structured analysis...]

============================================================
RESULTS COMPARISON
============================================================
Test Configuration: One-Sample two-tailed
Ground Truth p-value: 0.261564
OpenAI GPT-4 p-value: 0.26
DeepSeek p-value: 0.262

============================================================
ACCURACY ANALYSIS
============================================================
Hypotheses tested:
  H₀: μ = 10.0
  H₁: μ ≠ 10.0
Ground truth significant (α=0.05): False
OpenAI prediction: Not significant (p=0.26) - ✓ Correct
DeepSeek prediction: Not significant (p=0.262) - ✓ Correct
```

## 🔬 Prompt Variation Styles

### 1. Standard Prompt (Default)
Clear 4-section structure with explicit requirements:
- **Best for**: General use, baseline comparisons
- **Characteristics**: Direct instructions, structured format
- **Use case**: Standard hypothesis testing scenarios

### 2. Chain-of-Thought Prompt
Step-by-step reasoning with metacognitive checks:
- **Best for**: Complex analysis requiring transparent reasoning
- **Characteristics**: "First, Then, Finally" structure, explicit reasoning steps
- **Use case**: Educational demonstrations, debugging LLM reasoning
- **Research basis**: Wei et al. (2022) - Improves accuracy by 15-30%

### 3. Expert Role-Playing Prompt
Senior biostatistician perspective with professional standards:
- **Best for**: High-stakes analysis requiring rigor
- **Characteristics**: Professional language, peer-review standards, self-verification
- **Use case**: Research publications, clinical trials analysis
- **Research basis**: Shanahan et al. (2023) - Improves technical task performance

### 4. Structured Template Prompt
Visual formatting with error prevention warnings:
- **Best for**: Error-prone tasks needing systematic verification
- **Characteristics**: Visual dividers, checkboxes, explicit error warnings
- **Use case**: Quality control, automated systems
- **Research basis**: Liu et al. (2023) - Reduces errors by 30%

See [PROMPT_VARIATIONS.md](PROMPT_VARIATIONS.md) for detailed documentation.

## 📁 Project Structure

```
demo/
├── demo_2.0.py              # Main interactive demo with comprehensive features
├── demo_ttest.py            # Legacy version (basic functionality)
├── README.md                # This file
├── PROMPT_VARIATIONS.md     # Detailed prompt engineering documentation
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
└── wandb/                  # W&B experiment logs (offline mode)
```

## 🔧 Advanced Configuration

### Custom Prompts

You can provide your own prompts while maintaining the structured output format:

```python
# Example custom prompt
custom_prompt = """
As a biostatistician, analyze this clinical trial data.
Focus on practical significance alongside statistical significance.
Consider effect size and clinical relevance in your interpretation.
"""

# The system will automatically append:
# - Data
# - Null hypothesis
# - Significance level
# - Structured format requirements
# - JSON output format
```

### Data Constraints

Generate data within specific ranges for realistic scenarios:

```
Minimum value constraint (press Enter for none): 0
Maximum value constraint (press Enter for none): 20

→ All generated data will be constrained to [0, 20]
```

### Two-Sample Tests

Configure independent or paired sample comparisons:

```
1. SAMPLE COMPARISON TYPE:
   1. One-sample test
   2. Two-sample test
Select comparison type (1-2, default=1): 2

Sample size for first sample (default=30): 35
Sample size for second sample (default=30): 40

→ Generates two independent samples with Levene's test for variance equality
```

## 📊 Distribution Parameters

### Normal Distribution
```
Parameters:
  - Mean (μ): Center of distribution
  - Standard deviation (σ): Spread of distribution
  - Optional: Min/max constraints

Example: μ=10.0, σ=5.0, n=30
```

### Uniform Distribution
```
Parameters:
  - Low bound (a): Minimum value
  - High bound (b): Maximum value
  - Constraints automatically applied

Example: a=5.0, b=15.0, n=30
```

### Binomial Distribution
```
Parameters:
  - Number of trials (n): Trials per observation
  - Probability (p): Success probability
  - Optional: Min/max constraints

Example: n=20, p=0.5, sample_size=30
```

### Poisson Distribution
```
Parameters:
  - Lambda (λ): Rate parameter (mean)
  - Optional: Min/max constraints

Example: λ=15.0, sample_size=30
```

## 📈 Experiment Tracking

The platform automatically logs all experiments to Weights & Biases:

- **Configuration**: All test parameters and settings
- **Data**: Generated samples and summary statistics
- **Ground Truth**: SciPy statistical calculations
- **LLM Responses**: Full text and parsed results
- **Accuracy Metrics**: Comparison of LLM vs ground truth

**Offline Mode**: Experiments are saved locally in `wandb/` directory and can be synced later:
```bash
wandb sync wandb/offline-run-XXXXXX
```

## 🔍 Troubleshooting

### API Key Issues
```
Error: OpenAI/DeepSeek API key not set
Solution: Set environment variables as shown in API Keys Setup section
```

### Unicode Encoding Warnings
```
Warning: UnicodeEncodeError in W&B logging
Solution: The platform automatically sanitizes Unicode characters (μ→mu, ≠→!=)
Status: This is cosmetic and doesn't affect functionality
```

### JSON Parsing Failures
```
Issue: LLM response doesn't contain valid JSON
Solution: The platform extracts p-values from text if JSON parsing fails
Note: Both models typically provide properly formatted JSON
```

### Timeout Errors (DeepSeek)
```
Error: Read timed out
Solution: The platform has 90-second timeout; retry if server is busy
Note: DeepSeek may experience high traffic during peak hours
```

### Data Generation Constraints
```
Warning: Only generated X samples (requested Y) within constraints
Solution: Adjust min/max constraints or increase max_attempts parameter
Note: Very tight constraints may limit sample generation
```

## 🎓 Use Cases

### Educational Applications
- **Teaching Statistics**: Interactive demonstrations of hypothesis testing
- **LLM Literacy**: Understanding AI capabilities and limitations in quantitative reasoning
- **Prompt Engineering**: Experimenting with different instruction styles

### Research Applications
- **LLM Evaluation**: Systematic assessment of statistical reasoning abilities
- **Prompt Strategy Research**: Comparing effectiveness of different prompting approaches
- **Benchmarking**: Creating standardized tests for model comparison

### Development Applications
- **Quality Assurance**: Testing LLM outputs for statistical applications
- **Prompt Optimization**: Finding best instructions for specific use cases
- **Integration Testing**: Validating LLM integration in statistical workflows



## 🙏 Acknowledgments so far

- **SciPy**: Ground truth statistical calculations
- **OpenAI**: GPT-4 API access
- **DeepSeek**: DeepSeek Chat API access
- **Weights & Biases**: Experiment tracking platform
- **Research Papers**: Prompt engineering strategies from Wei et al. (2022), Shanahan et al. (2023), and Liu et al. (2023)
- **GitCopilot**: Code generator

--

**Last Updated**: November 2024  
**Python**: 3.8+  
**Status**: Active Development
