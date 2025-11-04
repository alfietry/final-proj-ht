"""
Interactive LLM Hypothesis Testing Demo - comparing GPT-4 and DeepSeek on statistical tasks.

Features:
  - Interactive terminal input for custom prompts and parameters
  - Multiple distribution types: Normal, Uniform, Binomial, Poisson
  - Configurable sample sizes and distribution parameters
  - Compares 2 LLM engines: OpenAI GPT-4, DeepSeek Chat
  - Ground truth calculation using scipy
  - Weights & Biases logging (offline mode)

Requirements (see requirements.txt):
  - openai
  - requests
  - scipy
  - numpy
  - wandb

Environment:
  - Set OPENAI_API_KEY to call OpenAI. If not set, the OpenAI call will be skipped.
  - Set DEEPSEEK_API_KEY to call DeepSeek. If not set, the DeepSeek call will be skipped.

Usage:
  python demo_2.0.py
  
The script will interactively prompt for:
  1. Custom prompt (optional)
  2. Distribution type selection
  3. Distribution parameters
  4. Sample size
"""
import os
import json
import math
import random
import statistics
import re
from typing import Optional, Dict, Any

import numpy as np
from scipy import stats

try:
    import openai
except Exception:
    openai = None

import requests
import wandb


import requests
import wandb


def generate_data(distribution_type, parameters, sample_size, min_constraint=None, max_constraint=None, seed=42):
    """Enhanced data generation function with constraint support."""
    np.random.seed(seed)
    
    # Generate initial sample
    if distribution_type == "normal":
        mean = parameters.get('mean', 10.0)
        std = parameters.get('std', 5.0)
        data = np.random.normal(mean, std, sample_size * 3)  # Generate extra to account for filtering
        
    elif distribution_type == "uniform":
        low = parameters.get('low', 5.0)
        high = parameters.get('high', 15.0)
        data = np.random.uniform(low, high, sample_size * 3)
        
    elif distribution_type == "binomial":
        n_trials = parameters.get('n_trials', 20)
        prob = parameters.get('prob', 0.5)
        data = np.random.binomial(n_trials, prob, sample_size * 3)
        
    elif distribution_type == "poisson":
        lam = parameters.get('lam', 10.0)
        data = np.random.poisson(lam, sample_size * 3)
    
    else:
        raise ValueError(f"Unsupported distribution: {distribution_type}")
    
    # Apply constraints if specified
    if min_constraint is not None or max_constraint is not None:
        if min_constraint is not None:
            data = data[data >= min_constraint]
        if max_constraint is not None:
            data = data[data <= max_constraint]
        
        # If we don't have enough data after filtering, generate more
        attempt_count = 0
        max_attempts = 100
        
        while len(data) < sample_size and attempt_count < max_attempts:
            attempt_count += 1
            
            # Generate additional data
            if distribution_type == "normal":
                additional_data = np.random.normal(mean, std, sample_size * 2)
            elif distribution_type == "uniform":
                additional_data = np.random.uniform(low, high, sample_size * 2)
            elif distribution_type == "binomial":
                additional_data = np.random.binomial(n_trials, prob, sample_size * 2)
            elif distribution_type == "poisson":
                additional_data = np.random.poisson(lam, sample_size * 2)
            
            # Apply constraints
            if min_constraint is not None:
                additional_data = additional_data[additional_data >= min_constraint]
            if max_constraint is not None:
                additional_data = additional_data[additional_data <= max_constraint]
            
            # Combine data
            data = np.concatenate([data, additional_data])
        
        if len(data) < sample_size:
            print(f"Warning: Only generated {len(data)} samples (requested {sample_size}) within constraints")
    
    # Take only the required number of samples and return as numpy array
    return data[:sample_size]


def generate_constrained_data(distribution_type, parameters, sample_size, seed=42, min_val=None, max_val=None, max_attempts=10000):
    """Generate data based on distribution type with min/max constraints."""
    if parameters is None:
        parameters = {}
    
    rng = np.random.default_rng(seed)
    
    # Generate more data than needed to allow for filtering
    oversample_factor = 3 if (min_val is not None or max_val is not None) else 1
    generate_size = sample_size * oversample_factor
    
    if distribution_type == "normal":
        mean = parameters.get('mean', 10.0)
        std = parameters.get('std', 5.0)
        data = rng.normal(loc=mean, scale=std, size=generate_size)
    
    elif distribution_type == "uniform":
        low = parameters.get('low', 5.0)
        high = parameters.get('high', 15.0)
        data = rng.uniform(low=low, high=high, size=generate_size)
    
    elif distribution_type == "binomial":
        n_trials = parameters.get('n_trials', 20)
        prob = parameters.get('prob', 0.5)
        data = rng.binomial(n=n_trials, p=prob, size=generate_size).astype(float)
    
    elif distribution_type == "poisson":
        lam = parameters.get('lam', 10.0)
        data = rng.poisson(lam=lam, size=generate_size).astype(float)
    
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
    
    # Apply constraints if specified
    if min_val is not None or max_val is not None:
        if min_val is not None:
            data = data[data >= min_val]
        if max_val is not None:
            data = data[data <= max_val]
        
        # If we don't have enough data after filtering, generate more
        attempts = 0
        max_attempts = 10
        while len(data) < sample_size and attempts < max_attempts:
            attempts += 1
            additional_data = generate_constrained_data(
                distribution_type, parameters, sample_size * oversample_factor, 
                seed + attempts, min_val, max_val
            )
            data = np.concatenate([data, additional_data])
        
        if len(data) < sample_size:
            print(f"Warning: Only generated {len(data)} samples (requested {sample_size}) within constraints")
    
    # Take only the required number of samples
    return data[:sample_size].tolist()


def generate_data_samples(config):
    """Generate data based on complete test configuration."""
    samples = {}
    
    if config['comparison_type'] == 'one-sample':
        samples['sample1'] = generate_constrained_data(
            distribution_type=config['distribution_type'],
            parameters=config['distribution_params'],
            sample_size=config['sample_size'],
            seed=config.get('seed', 42),
            min_val=config.get('min_constraint'),
            max_val=config.get('max_constraint')
        )
    
    elif config['comparison_type'] == 'two-sample':
        samples['sample1'] = generate_constrained_data(
            distribution_type=config['distribution_type'],
            parameters=config['distribution_params'],
            sample_size=config['sample_size'],
            seed=config.get('seed', 42),
            min_val=config.get('min_constraint'),
            max_val=config.get('max_constraint')
        )
        
        # Generate second sample with potentially different parameters
        sample2_params = config.get('distribution_params_2', config['distribution_params'].copy())
        # Slightly shift parameters for sample 2 to create difference
        if config['distribution_type'] == 'normal':
            sample2_params['mean'] = sample2_params.get('mean', 10.0) + 1.0
        elif config['distribution_type'] == 'uniform':
            sample2_params['low'] = sample2_params.get('low', 5.0) + 0.5
            sample2_params['high'] = sample2_params.get('high', 15.0) + 0.5
        elif config['distribution_type'] == 'poisson':
            sample2_params['lam'] = sample2_params.get('lam', 10.0) + 1.0
        
        samples['sample2'] = generate_constrained_data(
            distribution_type=config['distribution_type'],
            parameters=sample2_params,
            sample_size=config.get('sample_size_2', config['sample_size']),
            seed=config.get('seed', 42) + 1000,
            min_val=config.get('min_constraint'),
            max_val=config.get('max_constraint')
        )
    
    return samples


def auto_generate_alternate_hypothesis(null_hypothesis, test_type, tail_type):
    """Auto-generate alternate hypothesis from null hypothesis and test configuration."""
    if test_type == "one-sample":
        if "μ =" in null_hypothesis:
            mu_value = null_hypothesis.split("μ =")[1].strip()
            if tail_type == "two-tailed":
                return f"μ ≠ {mu_value}"
            elif tail_type == "left-tailed":
                return f"μ < {mu_value}"
            elif tail_type == "right-tailed":
                return f"μ > {mu_value}"
    
    elif test_type == "two-sample":
        if tail_type == "two-tailed":
            return "μ₁ ≠ μ₂"
        elif tail_type == "left-tailed":
            return "μ₁ < μ₂"
        elif tail_type == "right-tailed":
            return "μ₁ > μ₂"
    
    return "Alternative hypothesis could not be auto-generated"


def validate_hypothesis_consistency(null_hypothesis, alternate_hypothesis, tail_type):
    """Validate logical consistency between hypotheses and test type."""
    null_clean = null_hypothesis.replace(" ", "").lower()
    alt_clean = alternate_hypothesis.replace(" ", "").lower()
    
    # Check for consistency patterns
    if tail_type == "two-tailed":
        if "≠" not in alt_clean and "!=" not in alt_clean:
            return False, "Two-tailed test requires ≠ in alternate hypothesis"
    
    elif tail_type == "left-tailed":
        if "<" not in alt_clean:
            return False, "Left-tailed test requires < in alternate hypothesis"
    
    elif tail_type == "right-tailed":
        if ">" not in alt_clean:
            return False, "Right-tailed test requires > in alternate hypothesis"
    
    return True, "Hypotheses are consistent"


def get_user_inputs():
    """Complete hierarchical terminal menu system for statistical test configuration."""
    print("=" * 70)
    print("    COMPREHENSIVE LLM HYPOTHESIS TESTING CONFIGURATION")
    print("=" * 70)
    print()
    
    config = {}
    
    # 1. SAMPLE COMPARISON TYPE
    print("1. SAMPLE COMPARISON TYPE:")
    print("   1. One-sample test (compare sample mean to known value)")
    print("   2. Two-sample test (compare two sample means)")
    
    comparison_choice = input("Select comparison type (1-2, default=1): ").strip()
    config['comparison_type'] = 'two-sample' if comparison_choice == '2' else 'one-sample'
    print(f"   → Selected: {config['comparison_type'].upper()} test")
    print()
    
    # 2. TEST TAIL TYPE
    print("2. TEST TAIL TYPE:")
    print("   1. Two-tailed test (≠, checks for difference in either direction)")
    print("   2. Left-tailed test (<, checks if sample is significantly smaller)")
    print("   3. Right-tailed test (>, checks if sample is significantly larger)")
    
    tail_choice = input("Select tail type (1-3, default=1): ").strip()
    tail_mapping = {'1': 'two-tailed', '2': 'left-tailed', '3': 'right-tailed'}
    config['tail_type'] = tail_mapping.get(tail_choice, 'two-tailed')
    print(f"   → Selected: {config['tail_type'].upper()} test")
    print()
    
    # 3. HYPOTHESIS CONFIGURATION
    print("3. HYPOTHESIS CONFIGURATION:")
    
    # Null hypothesis input
    if config['comparison_type'] == 'one-sample':
        null_default = "μ = 10.0"
        null_input = input(f"Enter null hypothesis (default: {null_default}): ").strip()
        config['null_hypothesis'] = null_input if null_input else null_default
    else:
        null_default = "μ₁ = μ₂"
        null_input = input(f"Enter null hypothesis (default: {null_default}): ").strip()
        config['null_hypothesis'] = null_input if null_input else null_default
    
    print(f"   Null hypothesis (H₀): {config['null_hypothesis']}")
    
    # Alternate hypothesis handling
    auto_alt = auto_generate_alternate_hypothesis(
        config['null_hypothesis'], 
        config['comparison_type'], 
        config['tail_type']
    )
    
    print(f"   Auto-generated alternate: {auto_alt}")
    use_auto = input("Use auto-generated alternate hypothesis? (y/n, default=y): ").strip().lower()
    
    if use_auto in ['n', 'no']:
        alt_input = input("Enter custom alternate hypothesis: ").strip()
        config['alternate_hypothesis'] = alt_input if alt_input else auto_alt
    else:
        config['alternate_hypothesis'] = auto_alt
    
    print(f"   → Alternate hypothesis (H₁): {config['alternate_hypothesis']}")
    
    # Validate consistency
    is_valid, message = validate_hypothesis_consistency(
        config['null_hypothesis'], 
        config['alternate_hypothesis'], 
        config['tail_type']
    )
    
    if not is_valid:
        print(f"   ⚠️  Warning: {message}")
        proceed = input("   Continue anyway? (y/n, default=n): ").strip().lower()
        if proceed not in ['y', 'yes']:
            print("   Please restart and fix hypothesis configuration.")
            return None
    else:
        print(f"   ✓ {message}")
    
    print()
    
    # 4. PROMPT SELECTION (Updated)
    print("4. PROMPT SELECTION:")
    use_default_prompt = input("Use default prompt template? (y/n, default=y): ").strip().lower()
    
    if use_default_prompt in ['n', 'no']:
        print("Enter your custom prompt:")
        custom_prompt = input().strip()
        config['custom_prompt'] = custom_prompt if custom_prompt else None
        config['prompt_style'] = "standard"  # Custom prompts use standard structure
    else:
        config['custom_prompt'] = None
        
        # Add prompt style selection
        print("\n   Prompt Style Options:")
        print("   1. Standard (clear 4-section structure)")
        print("   2. Chain-of-Thought (step-by-step reasoning)")
        print("   3. Expert Role-Playing (senior biostatistician perspective)")
        print("   4. Structured Template (visual formatting with error prevention)")
        
        style_choice = input("   Select prompt style (1-4, default=1): ").strip()
        style_mapping = {
            '1': 'standard',
            '2': 'chain_of_thought',
            '3': 'expert',
            '4': 'template'
        }
        config['prompt_style'] = style_mapping.get(style_choice, 'standard')
        print(f"   → Using: {config['prompt_style'].replace('_', ' ').title()} prompt style")
    
    print()
    
    # 5. DISTRIBUTION SELECTION
    print("5. DATA GENERATION SETTINGS:")
    print("   Distribution types:")
    print("   1. Normal distribution")
    print("   2. Uniform distribution")
    print("   3. Binomial distribution")
    print("   4. Poisson distribution")
    
    dist_choice = input("Select distribution (1-4, default=1): ").strip()
    dist_mapping = {'1': 'normal', '2': 'uniform', '3': 'binomial', '4': 'poisson'}
    config['distribution_type'] = dist_mapping.get(dist_choice, 'normal')
    print(f"   → Selected: {config['distribution_type'].upper()} distribution")
    
    # Get distribution parameters with constraints
    config['distribution_params'] = {}
    print(f"\n   Parameters for {config['distribution_type'].upper()} distribution:")
    
    if config['distribution_type'] == "normal":
        mean_input = input("   Mean (default=10.0): ").strip()
        std_input = input("   Standard deviation (default=5.0): ").strip()
        config['distribution_params']['mean'] = float(mean_input) if mean_input else 10.0
        config['distribution_params']['std'] = float(std_input) if std_input else 5.0
        
    elif config['distribution_type'] == "uniform":
        low_input = input("   Low bound (default=5.0): ").strip()
        high_input = input("   High bound (default=15.0): ").strip()
        config['distribution_params']['low'] = float(low_input) if low_input else 5.0
        config['distribution_params']['high'] = float(high_input) if high_input else 15.0
        
    elif config['distribution_type'] == "binomial":
        n_trials_input = input("   Number of trials (default=20): ").strip()
        prob_input = input("   Probability of success (default=0.5): ").strip()
        config['distribution_params']['n_trials'] = int(n_trials_input) if n_trials_input else 20
        config['distribution_params']['prob'] = float(prob_input) if prob_input else 0.5
        
    elif config['distribution_type'] == "poisson":
        lam_input = input("   Lambda rate parameter (default=10.0): ").strip()
        config['distribution_params']['lam'] = float(lam_input) if lam_input else 10.0
    
    # Min/Max constraints
    print("\n   Data constraints (optional):")
    min_input = input("   Minimum value constraint (press Enter for none): ").strip()
    max_input = input("   Maximum value constraint (press Enter for none): ").strip()
    
    config['min_constraint'] = float(min_input) if min_input else None
    config['max_constraint'] = float(max_input) if max_input else None
    
    if config['min_constraint'] is not None or config['max_constraint'] is not None:
        print(f"   → Constraints: [{config['min_constraint'] or '-∞'}, {config['max_constraint'] or '+∞'}]")
    else:
        print("   → No constraints applied")
    
    print()
    
    # 6. SAMPLE SIZE SPECIFICATION
    print("6. SAMPLE SIZE SPECIFICATION:")
    sample_size_input = input("Sample size for first sample (default=30): ").strip()
    config['sample_size'] = int(sample_size_input) if sample_size_input else 30
    
    if config['comparison_type'] == 'two-sample':
        sample_size_2_input = input("Sample size for second sample (default=same as first): ").strip()
        config['sample_size_2'] = int(sample_size_2_input) if sample_size_2_input else config['sample_size']
        print(f"   → Sample sizes: {config['sample_size']} and {config['sample_size_2']}")
    else:
        print(f"   → Sample size: {config['sample_size']}")
    
    # Optional seed
    seed_input = input("Random seed (default=42): ").strip()
    config['seed'] = int(seed_input) if seed_input else 42
    print()
    
    # 7. CONFIGURATION REVIEW
    print("7. CONFIGURATION REVIEW:")
    print("   " + "=" * 50)
    print(f"   Test Type: {config['comparison_type'].title()} {config['tail_type']}")
    print(f"   Null Hypothesis: {config['null_hypothesis']}")
    print(f"   Alternate Hypothesis: {config['alternate_hypothesis']}")
    print(f"   Distribution: {config['distribution_type'].title()}")
    print(f"   Parameters: {config['distribution_params']}")
    if config['min_constraint'] is not None or config['max_constraint'] is not None:
        print(f"   Constraints: [{config['min_constraint']}, {config['max_constraint']}]")
    print(f"   Sample Size(s): {config['sample_size']}", end="")
    if config['comparison_type'] == 'two-sample':
        print(f", {config['sample_size_2']}")
    else:
        print()
    print(f"   Prompt: {'Custom' if config['custom_prompt'] else 'Default'}")
    print(f"   Random Seed: {config['seed']}")
    print("   " + "=" * 50)
    
    confirm = input("\nProceed with this configuration? (y/n, default=y): ").strip().lower()
    if confirm in ['n', 'no']:
        print("Configuration cancelled. Please restart.")
        return None
    
    print("\n✓ Configuration confirmed!")
    return config


def ground_truth_ttest(data, mu=10.0, alpha=0.05):
    """Calculate ground truth t-test statistics including critical values."""
    # Ensure data is a numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Flatten data in case it's multi-dimensional
    data = data.flatten()
    
    # Check if data is empty
    if len(data) == 0:
        raise ValueError("Data array is empty")
    
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)  # Sample standard deviation
    
    # Calculate t-statistic and p-value
    tstat, pvalue = stats.ttest_1samp(data, popmean=mu)
    
    # Calculate critical values (two-tailed)
    df = n - 1  # degrees of freedom
    critical_value = stats.t.ppf(1 - alpha/2, df)  # Two-tailed critical value
    
    return {
        'sample_mean': float(sample_mean),
        'sample_std': float(sample_std),
        'sample_size': n,
        'degrees_of_freedom': df,
        't_statistic': float(tstat),
        'p_value': float(pvalue),
        'critical_value_positive': float(critical_value),
        'critical_value_negative': float(-critical_value),
        'alpha': alpha,
        'null_hypothesis_mu': mu
    }


def display_enhanced_ground_truth(data, gt_result):
    """Display comprehensive ground truth analysis for one-sample test."""
    print("GROUND TRUTH ANALYSIS (One-Sample Test):")
    print(f"Sample Statistics:")
    print(f"  Sample mean: {gt_result['sample_mean']:.4f}")
    print(f"  Sample std dev: {gt_result['sample_std']:.4f}")
    print(f"  Sample size: {gt_result['sample_size']}")
    print(f"  Degrees of freedom: {gt_result['degrees_of_freedom']}")
    
    print(f"\nHypothesis Test:")
    print(f"  Null hypothesis: μ = {gt_result['null_hypothesis_mu']:.4f}")
    print(f"  Significance level: α = {gt_result['alpha']:.3f}")
    print(f"  T-statistic: {gt_result['t_statistic']:.4f}")
    print(f"  P-value: {gt_result['p_value']:.6f} (two-sided)")
    
    print(f"\nCritical Values (α = {gt_result['alpha']:.3f}):")
    print(f"  Critical values: ±{gt_result['critical_value_positive']:.3f}")
    print(f"  Lower critical: {gt_result['critical_value_negative']:.3f}")
    print(f"  Upper critical: {gt_result['critical_value_positive']:.3f}")
    
    decision = "Reject" if gt_result['p_value'] < gt_result['alpha'] else "Fail to reject"
    print(f"\nConclusion: {decision} H₀ at α = {gt_result['alpha']:.3f}")
    print()


def build_prompt(data, mu=10.0, alpha=0.05, custom_prompt=None, prompt_style="standard"):
    """Build a detailed prompt using various template styles or custom prompt.
    
    Args:
        data: Sample data (numpy array or list)
        mu: Null hypothesis mean value
        alpha: Significance level
        custom_prompt: User-provided custom prompt (overrides prompt_style)
        prompt_style: Template style - "standard", "chain_of_thought", "expert", or "template"
    """
    
    # Convert numpy array to list for JSON serialization
    data_list = data.tolist() if hasattr(data, 'tolist') else data
    
    # Style 1: Standard structured requirements
    STRUCTURED_REQUIREMENTS_STANDARD = """
IMPORTANT: You MUST follow this exact structure in your response:

1. TEST SELECTION: 
   - Which statistical test is appropriate and why?
   - Should this be a one-tailed or two-tailed test? Justify your choice.
   - State your alternative hypothesis (H₁) based on your tail selection.

2. CALCULATIONS: Show your work and calculations
   - Sample statistics (mean, standard deviation, sample size)
   - Test statistic calculation
   - Critical value(s) or p-value calculation

3. RESULTS: 
   - Test statistic value
   - P-value 
   - Critical value(s)
   - Confidence interval (if applicable)

4. CONCLUSION: State whether you reject or fail to reject H₀

Ensure your reasoning is statistically sound.

After completing the 4 sections above, end with a JSON summary:
{"p_value": <numeric_value>, "conclusion": "reject" or "fail to reject", "test_type": "one-tailed" or "two-tailed", "alternative_hypothesis": "your H1 statement"}
"""
    
    # Style 2: Chain-of-Thought with explicit reasoning
    STRUCTURED_REQUIREMENTS_COT = """
IMPORTANT: You MUST follow this exact structure and show your reasoning at each step:

1. TEST SELECTION:
   - First, identify the type of data and research question
   - Then, determine which statistical test is appropriate and explain why
   - Decide if this should be one-tailed or two-tailed based on the hypothesis direction
   - Finally, state your alternative hypothesis (H₁) that matches your tail selection
   - Reasoning check: Verify that your H₁ is logically consistent with the test type

2. CALCULATIONS (Show each step):
   - Step 2a: Calculate sample statistics
     * Sample mean: Show formula and computation
     * Sample standard deviation: Show formula and computation  
     * Sample size: Count and report
   - Step 2b: Calculate test statistic
     * State the formula for your chosen test
     * Substitute values into the formula
     * Compute the final test statistic value
   - Step 2c: Determine significance
     * Calculate or look up critical value(s) for α = 0.05
     * Calculate p-value using the test statistic
     * Show your work for both

3. RESULTS (Summarize your findings):
   - Test statistic value: [your calculated value]
   - P-value: [your calculated p-value]
   - Critical value(s): [for your chosen α level]
   - Confidence interval: [calculate 95% CI if applicable]
   - Verification: Compare p-value to α and test statistic to critical values

4. CONCLUSION:
   - Decision rule: State whether p-value < α (or |test statistic| > critical value)
   - Statistical decision: Clearly state "reject H₀" or "fail to reject H₀"
   - Practical interpretation: Explain what this means in the context of the hypothesis

Ensure your reasoning is statistically sound and each step logically follows from the previous.

After completing all 4 sections above, provide a JSON summary:
{"p_value": <numeric_value>, "conclusion": "reject" or "fail to reject", "test_type": "one-tailed" or "two-tailed", "alternative_hypothesis": "your H1 statement"}
"""
    
    # Style 3: Expert role-playing with self-verification
    STRUCTURED_REQUIREMENTS_EXPERT = """
IMPORTANT: You are a senior biostatistician reviewing a research analysis. Follow this structure and verify your work:

1. TEST SELECTION (Expert Analysis):
   - As a statistical expert, identify the appropriate test for this data
   - Explain your choice: Why is this test suitable for these conditions?
   - Determine tail type: Should this be one-tailed or two-tailed?
   - Justification: Provide statistical reasoning for your tail choice
   - State H₁: Formulate the alternative hypothesis based on your decisions
   - Self-check: Does your H₁ align with your tail type? (two-tailed uses ≠, one-tailed uses < or >)

2. CALCULATIONS (Show Professional-Grade Work):
   - Descriptive statistics:
     * Calculate and report: sample mean (x̄), standard deviation (s), sample size (n)
     * Verify: Check if assumptions are met (normality, independence, etc.)
   - Test statistic computation:
     * Formula: Write the appropriate formula for your test
     * Calculation: Substitute values and compute the statistic
     * Degrees of freedom: Calculate if applicable
   - Significance assessment:
     * Critical value(s): Determine for α = 0.05 and your tail type
     * P-value: Calculate the exact probability
     * Double-check: Ensure p-value matches your test statistic

3. RESULTS (Present Findings):
   - Test statistic: [report with appropriate precision]
   - P-value: [report to at least 3 decimal places]
   - Critical value(s): [report based on α = 0.05]
   - Confidence interval: [95% CI for the parameter if applicable]
   - Cross-validation: Confirm that statistical decision is consistent using both p-value and critical value approaches

4. CONCLUSION (Professional Interpretation):
   - Statistical decision: State whether you "reject H₀" or "fail to reject H₀"
   - Evidence strength: Describe the strength of evidence (e.g., p = 0.001 is strong, p = 0.04 is weak)
   - Context: Interpret the finding in terms of the original hypothesis
   - Peer review check: Is your conclusion logically supported by your results?

Ensure statistical rigor throughout your analysis. Verify each calculation before proceeding.

After completing the 4-section analysis, provide a JSON summary for automated processing:
{"p_value": <numeric_value>, "conclusion": "reject" or "fail to reject", "test_type": "one-tailed" or "two-tailed", "alternative_hypothesis": "your H1 statement"}
"""
    
    # Style 4: Structured template with error prevention
    STRUCTURED_REQUIREMENTS_TEMPLATE = """
CRITICAL INSTRUCTIONS: Follow this exact template. Each section is mandatory.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. TEST SELECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Statistical Test Choice:
  Test name: [specify the test]
  Reasoning: [explain why this test is appropriate for this scenario]
  
▶ Tail Type Determination:
  Choice: [one-tailed OR two-tailed]
  Justification: [explain based on the research question/hypothesis direction]
  
▶ Hypothesis Formulation:
  H₀ (Null): [already provided in the problem]
  H₁ (Alternative): [write using proper notation: μ ≠, μ <, or μ >]
  
⚠️ VERIFICATION: Confirm tail type matches H₁ symbol (two-tailed = ≠, left = <, right = >)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2. CALCULATIONS (Step-by-Step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Sample Statistics:
  • Sample mean (x̄) = [show calculation]
  • Sample std dev (s) = [show calculation]
  • Sample size (n) = [count]
  
▶ Test Statistic:
  • Formula: [write the formula]
  • Substitution: [plug in the values]
  • Result: [final computed value]
  
▶ Significance Values:
  • Critical value(s): [for α=0.05 and your tail type]
  • P-value calculation: [show method/steps]
  • P-value result: [final value between 0 and 1]

⚠️ COMMON ERRORS TO AVOID:
  - Don't confuse one-tailed and two-tailed p-values
  - Ensure degrees of freedom is n-1 for one-sample t-test
  - Verify critical values match your chosen α and tail type

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3. RESULTS (Summary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Test statistic value: [number]
  ✓ P-value: [number between 0 and 1]
  ✓ Critical value(s): [number(s)]
  ✓ Confidence interval: [if applicable, show 95% CI]
  
▶ Decision Criteria Check:
  • Is p-value < 0.05? [Yes/No]
  • Is |test statistic| > critical value? [Yes/No]
  • Do both methods agree? [Yes/No]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. CONCLUSION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
▶ Statistical Decision: 
  [Choose ONE: "Reject H₀" OR "Fail to reject H₀"]
  
▶ Reasoning:
  Because [explain using p-value comparison to α OR test statistic comparison to critical value]
  
▶ Interpretation:
  This means [explain what the decision implies about the hypothesis]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FINAL REQUIREMENT: After completing all sections, you MUST provide this JSON:
{"p_value": <exact_numeric_value>, "conclusion": "reject" or "fail to reject", "test_type": "one-tailed" or "two-tailed", "alternative_hypothesis": "your H1 statement"}
"""
    
    # Select the appropriate structured requirements based on style
    style_map = {
        "standard": STRUCTURED_REQUIREMENTS_STANDARD,
        "chain_of_thought": STRUCTURED_REQUIREMENTS_COT,
        "expert": STRUCTURED_REQUIREMENTS_EXPERT,
        "template": STRUCTURED_REQUIREMENTS_TEMPLATE
    }
    
    STRUCTURED_REQUIREMENTS = style_map.get(prompt_style, STRUCTURED_REQUIREMENTS_STANDARD)
    
    if custom_prompt:
        # Use custom prompt but ALWAYS include structured requirements and data
        return f"""{custom_prompt}

DATA: {json.dumps(data_list)}
NULL HYPOTHESIS (H₀): μ = {mu}
SIGNIFICANCE LEVEL (α): {alpha}

{STRUCTURED_REQUIREMENTS}"""
    
    # Default prompt template with structured requirements
    return f"""You are an expert statistician. Perform a hypothesis test with the following information:

DATA: {json.dumps(data_list)}
NULL HYPOTHESIS (H₀): μ = {mu}
SIGNIFICANCE LEVEL (α): {alpha}

{STRUCTURED_REQUIREMENTS}"""


def call_openai(prompt: str, model: str = "gpt-4") -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or openai is None:
        print("OpenAI API key not set or openai package missing — skipping OpenAI call.")
        return None
    
    messages = [
        {"role": "system", "content": (
            "You are an expert statistician. Provide detailed statistical analysis "
            "followed by a JSON summary as requested in the prompt."
        )},
        {"role": "user", "content": prompt},
    ]

    def _extract_content(resp_obj):
        # Try several common response shapes
        try:
            # new client: resp.choices[0].message.content
            return resp_obj.choices[0].message.content.strip()
        except Exception:
            pass
        try:
            # dict-like
            return resp_obj["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        try:
            # older openai: resp.choices[0].message['content'] or text
            return resp_obj.choices[0].text.strip()
        except Exception:
            pass
        # last resort: stringified response
        return str(resp_obj)

    try:
        # If the package exposes OpenAI (modern client), use it
        OpenAIClient = getattr(openai, "OpenAI", None)
        if OpenAIClient is not None:
            client = OpenAIClient()
            resp = client.chat.completions.create(
                model=model, 
                messages=messages, 
                max_tokens=800,  # Increased from 300
                temperature=0.0
            )
        else:
            # fallback to legacy interface (may raise on openai>=1.0)
            # keep compatibility for older installed versions
            try:
                openai.api_key = api_key
            except Exception:
                pass
            resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=300, temperature=0.0)

        # Log raw response for debugging (W&B will capture stdout)
        try:
            print("OPENAI_RAW_RESPONSE_JSON:", json.dumps(resp, default=str))
        except Exception:
            print("OPENAI_RAW_RESPONSE (repr):", repr(resp))

        content = _extract_content(resp)
        return content
    except Exception as e:
        print(f"OpenAI call failed: {e}")
        return None


def call_ollama(prompt: str, model: str = "gemma3:4b") -> Optional[str]:
    # Ollama local API assumed at http://localhost:11434
    base = "http://localhost:11434/api"
    gen_url = f"{base}/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 200}}
    try:
        r = requests.post(gen_url, json=payload, timeout=120)
        if r.status_code == 404:
            # Fallback to chat API if generate not available
            chat_url = f"{base}/chat"
            chat_payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.0, "num_predict": 200}}
            rc = requests.post(chat_url, json=chat_payload, timeout=120)
            rc.raise_for_status()
            data = rc.json()
            msg = (data.get("message") or {}).get("content")
            return msg.strip() if isinstance(msg, str) else json.dumps(data)
        r.raise_for_status()
        data = r.json()
        # Non-streaming generate returns 'response'
        if isinstance(data, dict) and isinstance(data.get("response"), str):
            return data["response"].strip()
        return json.dumps(data)
    except Exception as e:
        print(f"Ollama call failed or Ollama not running: {e}")
        return None


def call_deepseek(prompt: str, model: str = "deepseek-chat") -> Optional[str]:
    """Call DeepSeek Chat API for hypothesis testing analysis."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DeepSeek API key not set — skipping DeepSeek call.")
        return None
    
    print(f"Using DeepSeek model: {model}")  # Debug info
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": (
            "You are an expert statistician. Provide detailed statistical analysis "
            "followed by a JSON summary as requested in the prompt. "
            "Follow the exact format: 1. TEST SELECTION, 2. CALCULATIONS, 3. RESULTS, 4. CONCLUSION, then JSON."
        )},
        {"role": "user", "content": prompt},
    ]
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 800,  # Slightly reduced
        "temperature": 0.0,
        "stream": False
    }
    
    try:
        r = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=90  # Increased timeout to 90 seconds
        )
        
        print(f"DeepSeek HTTP Status: {r.status_code}")  # Debug info
        
        r.raise_for_status()
        data = r.json()
        
        # Log raw response for debugging
        try:
            print("DEEPSEEK_RAW_RESPONSE_JSON:", json.dumps(data, default=str))
        except Exception:
            print("DEEPSEEK_RAW_RESPONSE (repr):", repr(data))
        
        # Extract content from response
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0]["message"]["content"].strip()
            return content
        
        return None
    except requests.exceptions.Timeout:
        print("DeepSeek call timed out - server may be busy, try again later")
        return None
    except Exception as e:
        print(f"DeepSeek call failed: {e}")
        return None


def parse_json_pvalue(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Remove common wrappers (code fences) and trim
    cleaned = re.sub(r"```.*?```", "", text, flags=re.S).strip()

    # 1) Try to find a JSON object
    m_json = re.search(r"\{[\s\S]*\}", cleaned)
    if m_json:
        try:
            j = json.loads(m_json.group(0))
            p = None
            conclusion = None
            test_type = None
            alternative_hypothesis = None
            
            for k, v in j.items():
                kl = k.lower()
                if kl.startswith('p') and p is None:
                    try:
                        p = float(v)
                    except Exception:
                        # sometimes value is '≈0.12' or similar
                        vs = str(v).lstrip('≈~ ').strip()
                        try:
                            p = float(vs)
                        except Exception:
                            pass
                if isinstance(v, str) and ('reject' in v.lower() or 'fail' in v.lower()):
                    conclusion = v
                if 'test_type' in kl or 'tail' in kl:
                    test_type = v
                if 'alternative' in kl or 'h1' in kl or 'h_1' in kl:
                    alternative_hypothesis = v
                    
            return {
                "p_value": p, 
                "conclusion": conclusion, 
                "test_type": test_type,
                "alternative_hypothesis": alternative_hypothesis,
                "raw": j
            }
        except Exception:
            pass

    # 2) Try to find explicit p-value patterns like 'p = 0.123', 'p-value:0.12', or '≈0.12'
    patterns = [
        r"p-value\s*[:=≈]\s*([0-9]*\.?[0-9]+([eE][+-]?\d+)?)",  # Most specific first
        r"p\s*[:=]\s*≈?\s*([0-9]*\.?[0-9]+([eE][+-]?\d+)?)",    # General p-value
        r"p-value\s*\w*\s*([0-9]*\.?[0-9]+)",                   # p-value with words
        ]
    for pat in patterns:
        m = re.search(pat, cleaned, flags=re.I)
        if m:
            try:
                pv = float(m.group(1))
                # Only accept reasonable p-values (between 0 and 1)
                if 0.0 <= pv <= 1.0:
                    return {"p_value": pv, "conclusion": None}
            except Exception:
                continue

    # 3) Fallback: find any float-like token between 0 and 1
    tokens = re.findall(r"[0-9]*\.?[0-9]+([eE][+-]?\d+)?", cleaned)
    # re.findall above returns capture groups; better to find all numeric substrings
    num_tokens = re.findall(r"[0-9]+\.?[0-9]*([eE][+-]?\d+)?|\.[0-9]+([eE][+-]?\d+)?", cleaned)
    # flatten and re-find floats in cleaned text
    float_candidates = re.findall(r"[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?", cleaned)
    for tok in float_candidates:
        try:
            v = float(tok)
            if 0.0 <= v <= 1.0:
                return {"p_value": v, "conclusion": None}
        except Exception:
            continue

    return None


def perform_two_sample_test(data1, data2, test_config):
    """Perform appropriate two-sample statistical test based on configuration."""
    from scipy import stats
    
    # Determine if we should assume equal variances
    # Using Levene's test to check variance equality
    stat, p_val = stats.levene(data1, data2)
    equal_var = p_val > 0.05  # Assume equal variance if p > 0.05
    
    tail_type = test_config['tail_type']
    
    # Perform t-test
    if tail_type == 'two-tailed':
        alternative = 'two-sided'
    elif tail_type == 'left-tailed':
        alternative = 'less'
    elif tail_type == 'right-tailed':
        alternative = 'greater'
    else:
        alternative = 'two-sided'
    
    # Perform independent t-test
    t_stat, p_value = stats.ttest_ind(data1, data2, equal_var=equal_var, alternative=alternative)
    
    # Calculate additional statistics
    mean1, mean2 = np.mean(data1), np.mean(data2)
    std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
    
    # Degrees of freedom calculation
    if equal_var:
        # Pooled variance case
        n1, n2 = len(data1), len(data2)
        pooled_var = ((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2)
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
        df = n1 + n2 - 2
    else:
        # Welch's t-test case
        se1, se2 = std1/np.sqrt(len(data1)), std2/np.sqrt(len(data2))
        se_diff = np.sqrt(se1**2 + se2**2)
        df = (se1**2 + se2**2)**2 / (se1**4/(len(data1)-1) + se2**4/(len(data2)-1))
    
    # Critical value
    if tail_type == 'two-tailed':
        alpha = 0.05
        critical_value = stats.t.ppf(1 - alpha/2, df)
    else:
        alpha = 0.05
        critical_value = stats.t.ppf(1 - alpha, df)
    
    return {
        "test_type": "two_sample_t_test",
        "t_statistic": t_stat,
        "p_value": p_value,
        "degrees_freedom": df,
        "critical_value": critical_value,
        "sample1_mean": mean1,
        "sample1_std": std1,
        "sample1_size": len(data1),
        "sample2_mean": mean2,
        "sample2_std": std2,
        "sample2_size": len(data2),
        "mean_difference": mean1 - mean2,
        "standard_error_diff": se_diff,
        "equal_variances": equal_var,
        "alternative": alternative
    }


def build_two_sample_prompt(data1, data2, config, custom_prompt=None):
    """Build prompt for two-sample hypothesis testing."""
    
    if custom_prompt:
        return custom_prompt
    
    null_hyp = config['null_hypothesis']
    alt_hyp = config['alternate_hypothesis']
    tail_type = config['tail_type']
    
    prompt = f"""
You are a statistical analyst. Given two samples of data, perform a two-sample t-test to test the following hypotheses:

Null hypothesis (H₀): {null_hyp}
Alternative hypothesis (H₁): {alt_hyp}

Test type: {tail_type}

Sample 1 data: {data1.tolist()}
Sample 2 data: {data2.tolist()}

Please:
1. Calculate the appropriate test statistic
2. Determine the p-value for this {tail_type} test
3. State your conclusion at α = 0.05 significance level

Provide your response in the following JSON format:
{{
    "test_statistic": <calculated t-statistic>,
    "p_value": <calculated p-value>,
    "conclusion": "<reject or fail to reject H₀>",
    "reasoning": "<brief explanation of your calculation>"
}}

Ensure the p-value is calculated correctly for the specified test type ({tail_type}).
"""
    return prompt


def main():
    # Get comprehensive user configuration
    config = get_user_inputs()
    
    if config is None:
        print("Configuration failed or cancelled.")
        return None
    
    # Generate data based on test type
    print("\n" + "="*60)
    print("DATA GENERATION")
    print("="*60)
    
    if config['comparison_type'] == 'one-sample':
        # Generate single sample
        data = generate_data(
            distribution_type=config['distribution_type'], 
            parameters=config['distribution_params'], 
            sample_size=config['sample_size'],
            min_constraint=config['min_constraint'],
            max_constraint=config['max_constraint'],
            seed=config['seed']
        )
        
        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        print(f"Generated data type: {type(data)}, shape: {data.shape}")  # Debug info
        
        # Extract null hypothesis mean for one-sample test
        if "μ =" in config['null_hypothesis']:
            mu_str = config['null_hypothesis'].split("μ =")[1].strip()
            try:
                mu = float(mu_str)
            except ValueError:
                print(f"Warning: Could not parse mu from '{mu_str}', using default 10.0")
                mu = 10.0
        else:
            mu = 10.0  # default
        
        try:
            # Compute ground truth for one-sample test
            gt_result = ground_truth_ttest(data, mu=mu, alpha=0.05)
            
            # Display enhanced ground truth
            display_enhanced_ground_truth(data, gt_result)
            
        except Exception as e:
            print(f"Error in ground truth calculation: {e}")
            print(f"Data type: {type(data)}, Data shape: {data.shape if hasattr(data, 'shape') else 'No shape'}")
            print(f"Data sample: {data[:5] if len(data) > 0 else 'Empty data'}")
            return None
        
        # Build prompt with selected style
        prompt = build_prompt(
            data, 
            mu=mu, 
            alpha=0.05, 
            custom_prompt=config['custom_prompt'],
            prompt_style=config.get('prompt_style', 'standard')  # Use selected style
        )
        
        print(f"Generated one sample of size {len(data)}")
        
    elif config['comparison_type'] == 'two-sample':
        # Generate two samples
        data1 = generate_data(
            distribution_type=config['distribution_type'], 
            parameters=config['distribution_params'], 
            sample_size=config['sample_size'],
            min_constraint=config['min_constraint'],
            max_constraint=config['max_constraint'],
            seed=config['seed']
        )
        
        # Generate second sample with different seed to ensure independence
        data2 = generate_data(
            distribution_type=config['distribution_type'], 
            parameters=config['distribution_params'], 
            sample_size=config['sample_size_2'],
            min_constraint=config['min_constraint'],
            max_constraint=config['max_constraint'],
            seed=config['seed'] + 1000  # Different seed
        )
        
        # Compute ground truth for two-sample test
        gt_result = perform_two_sample_test(data1, data2, config)
        
        # Display two-sample ground truth
        print("GROUND TRUTH ANALYSIS (Two-Sample Test):")
        print(f"Sample 1 - Mean: {gt_result['sample1_mean']:.4f}, Std: {gt_result['sample1_std']:.4f}, Size: {gt_result['sample1_size']}")
        print(f"Sample 2 - Mean: {gt_result['sample2_mean']:.4f}, Std: {gt_result['sample2_std']:.4f}, Size: {gt_result['sample2_size']}")
        print(f"Mean difference (Sample1 - Sample2): {gt_result['mean_difference']:.4f}")
        print(f"Standard error of difference: {gt_result['standard_error_diff']:.4f}")
        print(f"T-statistic: {gt_result['t_statistic']:.4f}")
        print(f"Degrees of freedom: {gt_result['degrees_freedom']:.2f}")
        print(f"Critical value: ±{gt_result['critical_value']:.4f}")
        print(f"P-value: {gt_result['p_value']:.6f}")
        print(f"Equal variances assumed: {gt_result['equal_variances']}")
        
        # Build two-sample prompt
        prompt = build_two_sample_prompt(data1, data2, config, config['custom_prompt'])
        
        print(f"Generated two samples of sizes {len(data1)} and {len(data2)}")
    
    print()
    
    # Initialize W&B with comprehensive configuration
    os.environ.setdefault('WANDB_MODE', 'offline')
    
    # Sanitize Unicode characters for W&B logging to avoid encoding errors
    def sanitize_for_wandb(text):
        """Replace Unicode characters that cause W&B logging issues."""
        if not isinstance(text, str):
            return text
        replacements = {
            'μ': 'mu',
            '₁': '1', 
            '₂': '2',
            '≠': '!=',
            '≥': '>=',
            '≤': '<=',
            'α': 'alpha'
        }
        for unicode_char, ascii_char in replacements.items():
            text = text.replace(unicode_char, ascii_char)
        return text
    
    run = wandb.init(
        project="llm-hypothesis-testing-comprehensive",
        config={
            "comparison_type": config['comparison_type'],
            "tail_type": config['tail_type'],
            "distribution": config['distribution_type'],
            "distribution_params": config['distribution_params'],
            "sample_size": config['sample_size'],
            "sample_size_2": config.get('sample_size_2'),
            "null_hypothesis": sanitize_for_wandb(config['null_hypothesis']),
            "alternate_hypothesis": sanitize_for_wandb(config['alternate_hypothesis']),
            "constraints": {
                "min": config['min_constraint'],
                "max": config['max_constraint']
            },
            "custom_prompt": config['custom_prompt'] is not None,
            "seed": config['seed'],
            "ground_truth_p_value": float(gt_result["p_value"])  # Ensure it's a Python float, not numpy
        },
        mode="offline"
    )
    
    # Test LLM models
    print("="*60)
    print("TESTING LLM RESPONSES")
    print("="*60)
    
    results = {}
    
    # Test OpenAI GPT-4
    print("Testing OpenAI GPT-4...")
    openai_response = call_openai(prompt)
    
    # Display full OpenAI response
    if openai_response:
        print("\n" + "="*50)
        print("OPENAI GPT-4 FULL RESPONSE:")
        print("="*50)
        print(openai_response)
        print("="*50)
    
    openai_p_value = parse_json_pvalue(openai_response)
    results['openai'] = {
        'response': openai_response,
        'parsed_p_value': openai_p_value
    }
    
    # Test DeepSeek
    print("Testing DeepSeek...")
    deepseek_response = call_deepseek(prompt)
    
    # Display full DeepSeek response
    if deepseek_response:
        print("\n" + "="*50)
        print("DEEPSEEK FULL RESPONSE:")
        print("="*50)
        print(deepseek_response)
        print("="*50)
    
    deepseek_p_value = parse_json_pvalue(deepseek_response)
    results['deepseek'] = {
        'response': deepseek_response,
        'parsed_p_value': deepseek_p_value
    }
    
    # Log comprehensive results to W&B
    wandb.log({
        "openai_response": openai_response,
        "openai_parsed_p_value": openai_p_value,
        "deepseek_response": deepseek_response,
        "deepseek_parsed_p_value": deepseek_p_value,
        "ground_truth_p_value": gt_result["p_value"],
        "test_type": config['comparison_type'],
        "tail_type": config['tail_type']
    })
    
    # Display comprehensive results
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"Test Configuration: {config['comparison_type'].title()} {config['tail_type']}")
    print(f"Ground Truth p-value: {gt_result['p_value']:.6f}")
    
    # Display parsed p-values more clearly
    openai_pval_display = "None"
    if openai_p_value is not None and isinstance(openai_p_value, dict):
        openai_pval_display = f"{openai_p_value.get('p_value', 'None')}"
    elif openai_p_value is not None:
        openai_pval_display = f"{openai_p_value}"
        
    deepseek_pval_display = "None"
    if deepseek_p_value is not None and isinstance(deepseek_p_value, dict):
        deepseek_pval_display = f"{deepseek_p_value.get('p_value', 'None')}"
    elif deepseek_p_value is not None:
        deepseek_pval_display = f"{deepseek_p_value}"
    
    print(f"OpenAI GPT-4 p-value: {openai_pval_display}")
    print(f"DeepSeek p-value: {deepseek_pval_display}")
    
    # Comprehensive accuracy analysis
    print(f"\n{'='*60}")
    print("ACCURACY ANALYSIS")
    print(f"{'='*60}")
    
    alpha = 0.05
    gt_significant = gt_result['p_value'] < alpha
    
    print(f"Hypotheses tested:")
    print(f"  H₀: {config['null_hypothesis']}")
    print(f"  H₁: {config['alternate_hypothesis']}")
    print(f"Ground truth significant (α={alpha}): {gt_significant}")
    
    # Extract p-values from parsed results
    openai_pval = None
    if openai_p_value is not None:
        if isinstance(openai_p_value, dict):
            openai_pval = openai_p_value.get('p_value')
        else:
            openai_pval = openai_p_value
    
    deepseek_pval = None  
    if deepseek_p_value is not None:
        if isinstance(deepseek_p_value, dict):
            deepseek_pval = deepseek_p_value.get('p_value')
        else:
            deepseek_pval = deepseek_p_value
    
    # Analyze OpenAI results
    if openai_pval is not None and isinstance(openai_pval, (int, float)):
        openai_significant = openai_pval < alpha
        openai_correct = openai_significant == gt_significant
        print(f"OpenAI prediction: {'Significant' if openai_significant else 'Not significant'} (p={openai_pval:.3f}) - {'✓ Correct' if openai_correct else '✗ Incorrect'}")
    else:
        print("OpenAI: Failed to parse p-value")
    
    # Analyze DeepSeek results
    if deepseek_pval is not None and isinstance(deepseek_pval, (int, float)):
        deepseek_significant = deepseek_pval < alpha
        deepseek_correct = deepseek_significant == gt_significant
        print(f"DeepSeek prediction: {'Significant' if deepseek_significant else 'Not significant'} (p={deepseek_pval:.3f}) - {'✓ Correct' if deepseek_correct else '✗ Incorrect'}")
    else:
        print("DeepSeek: Failed to parse p-value")
    
    # Finish W&B run
    wandb.finish()
    
    print(f"\n{'='*60}")
    print("COMPREHENSIVE DEMO COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    main()
