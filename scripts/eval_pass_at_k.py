"""
Pass@k Evaluation Script
Evaluates model performance at different k values (1, 2, 4, 8, 16)

Usage:
    # Basic usage (with VLLMEngine)
    python scripts/eval_pass_at_k.py \
        --engine_name qwen3_vl_4b_i \
        --dataset_name kdeng03/mol-rep-ocr-v1.1 \
        --dataset_split valid \
        --max_samples 100
    
    # With custom sampling parameters
    python scripts/eval_pass_at_k.py \
        --engine_name qwen3_vl_4b_i \
        --dataset_name kdeng03/mol-rep-ocr-v1.1 \
        --max_k 16 \
        --temperature 0.7 \
        --top_p 0.9 \
        --max_new_tokens 256
    
    # Save results
    python scripts/eval_pass_at_k.py \
        --engine_name qwen3_vl_4b_i \
        --dataset_name kdeng03/mol-rep-ocr-v1.1 \
        --output_file results/pass_at_k_results.json
"""

import os
import sys
import json
import logging
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_TOKEN"] = "your_key"

from PIL import Image
from datasets import load_dataset

# import sys
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import compute_mol_metrics
from src.inference.inference_engine_vllm import VLLMEngine

# Import GRPO reward function (interface for future replacement)
try:
    from scripts.grpo_trl import comprehensive_reward_func as grpo_reward_func
    HAS_GRPO_REWARD = True
except ImportError:
    HAS_GRPO_REWARD = False
    grpo_reward_func = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)


# ==================== Reward Calculator (Pluggable) ====================

class RewardCalculator:
    """
    Reward calculator, supports importing from grpo_trl or custom implementations
    
    Usage:
        # Use grpo_trl reward
        calc = RewardCalculator()
        
        # Custom reward
        def my_reward(completion, gt_rep, rep_type):
            results = compute_mol_metrics(completion, gt_rep, rep_type)
            return 1.0 if results["is_pred_valid"] else 0.0
        
        calc = RewardCalculator(reward_func=my_reward)
    """
    
    def __init__(self, reward_func=None):
        self.reward_func = reward_func or self._default_reward_func
    
    def compute_single_reward(self, completion: str, gt_rep: str, rep_type: str) -> float:
        """Compute reward for a single completion"""
        return self.reward_func(completion, gt_rep, rep_type)
    
    def compute_group_rewards(self, completions: List[str], gt_rep: str, rep_type: str) -> List[float]:
        """Compute rewards for a group of completions"""
        return [self.compute_single_reward(c, gt_rep, rep_type) for c in completions]
    
    def _default_reward_func(self, completion: str, gt_rep: str, rep_type: str) -> float:
        """
        Default reward function (consistent with grpo_trl.py)
        
        Reward = 1.0 (valid) + 1.5 (inchikey_match) + 0.5 (tanimoto_sim)
        Range: 0.0 ~ 3.0
        """
        results = compute_mol_metrics(completion, gt_rep, rep_type)
        if not results["is_pred_valid"]:
            return 0.0
        return 1.0 + 1.5 * int(results["is_inchikey_match"]) + 0.5 * float(results["tanimoto_sim"])


# ==================== Sample Difficulty Classifier (Strategy A: pass@k) ====================

class SampleDifficultyClassifier:
    """
    Sample difficulty classifier based on pass@k behavior
    
    Classification strategy:
        - easy:      pass at k=1 (model outputs stably)
        - medium:    pass at 1 < k <= max_k/2 (requires some sampling)
        - hard:      pass at k > max_k/2 (requires extensive sampling)
        - impossible: no pass at k=max_k (may need data cleaning)
    
    Usage:
        classifier = SampleDifficultyClassifier(max_k=16)
        difficulty = classifier.classify_sample(rewards=[0.0, 1.5, 2.0, ...])
        # -> "medium" (first pass at k=2)
    """
    
    def __init__(self, max_k: int = 16):
        self.max_k = max_k
    
    def classify_sample(self, sample: Dict) -> str:
        """
        Classify sample difficulty based on results
        
        Args:
            sample: Sample dictionary containing "results"
            
        Returns:
            "easy", "medium", "hard", or "impossible"
        """
        results = sample["results"]
        for i, r in enumerate(results):
            if r.get("is_inchikey_match", False):  # InChIKey match counts as pass
                first_pass_k = i + 1
                if first_pass_k == 1:
                    return "easy"
                elif first_pass_k <= self.max_k // 2:
                    return "medium"
                else:
                    return "hard"
        return "impossible"
    
    def classify_dataset(self, all_samples: List[Dict]) -> Dict[str, int]:
        """
        Compute difficulty distribution across the dataset
        
        Args:
            all_samples: List of result dictionaries for each sample
            
        Returns:
            {"easy": 100, "medium": 50, "hard": 30, "impossible": 20}
        """
        distribution = {"easy": 0, "medium": 0, "hard": 0, "impossible": 0}
        for sample in all_samples:
            difficulty = self.classify_sample(sample)
            distribution[difficulty] += 1
        return distribution


class PassAtKVLLMEngine(VLLMEngine):
    """
    Extends VLLMEngine with pass@k evaluation methods
    
    Core features:
        - Generate n different completions at once
        - Supports image-text and text-only
        - Reuses parent class config, LoRA, and build logic
        - Supports full-batch and chunked-batch
    """
    
    def generate_n_completions(
        self,
        sample: dict,
        n: int = 16,
        prompt_key: str = "prompt",
        image_key: str = "input_rep",
        **sampling_overrides,
    ) -> List[str]:
        """
        Generate n different completions for a single sample (single-sample mode)
        
        Args:
            sample: Single sample from HF dataset
            n: Number of generations
            prompt_key: Prompt field name
            image_key: Image field name
            **sampling_overrides: Override default sampling parameters
            
        Returns:
            List of n generated texts
        """
        if self.llm is None:
            self.build()
        
        # Construct messages
        question = sample[prompt_key]
        image = sample.get(image_key)
        
        messages = []
        # Determine if VLM task: image must be PIL.Image type
        is_image = isinstance(image, Image.Image)
        
        if is_image:
            # VLM image-text task
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": image},
                    {"type": "text", "text": question},
                ],
            })
        else:
            # Text-only task (image is str or None)
            messages.append({"role": "user", "content": question})
        
        # Set sampling parameters, key: n=k
        sampling_params = self._get_default_sampling_params(n=n, **sampling_overrides)
        
        # Call parent class chat, return_text=False to get full output
        outputs = self.chat(messages, sampling_params=sampling_params, return_text=False)
        
        # Extract all n completions
        # outputs[0] is RequestOutput for the first prompt
        # outputs[0].outputs contains n CompletionOutputs
        completions = [output.text for output in outputs[0].outputs]
        
        return completions
    
    def generate_n_completions_batch(
        self,
        samples: List[dict],
        n: int = 16,
        prompt_key: str = "prompt",
        image_key: str = "input_rep",
        **sampling_overrides,
    ) -> List[List[str]]:
        """
        Batch generate n different completions for multiple samples (batch mode)
        
        Args:
            samples: List of samples from HF dataset
            n: Number of generations per sample
            prompt_key: Prompt field name
            image_key: Image field name
            **sampling_overrides: Override default sampling parameters
            
        Returns:
            List of n generated texts per sample: [[comp1, comp2, ...], [comp1, comp2, ...], ...]
        """
        if self.llm is None:
            self.build()
        
        # Construct batch messages
        all_messages = []
        for sample in samples:
            question = sample[prompt_key]
            image = sample.get(image_key)
            
            # Determine if VLM task: image must be PIL.Image type
            is_image = isinstance(image, Image.Image)
            
            if is_image:
                # VLM image-text task
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image},
                        {"type": "text", "text": question},
                    ],
                }]
            else:
                # Text-only task (image is str or None)
                messages = [{"role": "user", "content": question}]
            
            all_messages.append(messages)
        
        # Set sampling parameters, key: n=k
        sampling_params = self._get_default_sampling_params(n=n, **sampling_overrides)
        
        # Batch call vLLM chat
        outputs = self.chat(all_messages, sampling_params=sampling_params, return_text=False)
        
        # Extract all completions
        all_completions = []
        for output in outputs:
            completions = [o.text for o in output.outputs]
            all_completions.append(completions)
        
        return all_completions


@dataclass
class EvalArguments:
    """Evaluation argument configuration"""
    # Required fields (no defaults) must come first
    engine_name: str = field(
        metadata={"help": "VLLMEngine config name (in configs/model_inference_schema.yaml)"}
    )
    dataset_name: str = field(
        metadata={"help": "HuggingFace dataset name"}
    )
    # Optional fields (with defaults) come after
    config_path: str = field(
        default="configs/model_inference_schema.yaml",
        metadata={"help": "Inference config file path"}
    )
    dataset_split: str = field(
        default="valid",
        metadata={"help": "Dataset split (train, valid, test, extra_train, extra_valid)"}
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "Maximum evaluation samples, -1 means use all"}
    )
    max_k: int = field(
        default=16,
        metadata={"help": "Maximum k value, automatically tests 1,2,4,...,max_k"}
    )
    max_new_tokens: int = field(
        default=256,
        metadata={"help": "Maximum tokens to generate"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=0.9,
        metadata={"help": "Nucleus sampling top_p parameter"}
    )
    output_file: str = field(
        default=None,
        metadata={"help": "Output file path (.json or .csv)"}
    )
    prompt_key: str = field(
        default="prompt",
        metadata={"help": "Prompt field key in dataset"}
    )
    image_key: str = field(
        default="input_rep",
        metadata={"help": "Image field key in dataset, None for text-only"}
    )
    infer_batch_size: int = field(
        default=-1,
        metadata={"help": "Inference batch size, -1 for full batch, >0 for chunked batches"}
    )


def get_k_values(max_k: int) -> List[int]:
    """Generate k values list: 1, 2, 4, 8, ..., max_k"""
    k_values = []
    k = 1
    while k <= max_k:
        k_values.append(k)
        k *= 2
    return k_values


def evaluate_pass_at_k(
    engine: PassAtKVLLMEngine,
    dataset,
    args: EvalArguments
) -> Dict[str, Any]:
    """Evaluate pass@k metrics"""
    k_values = get_k_values(args.max_k)
    max_k = max(k_values)
    
    logger.info(f"🎯 Evaluating pass@k, k values: {k_values}")
    logger.info(f"📊 Samples: {len(dataset)}")
    logger.info(f"📦 Inference batch size: {'full batch' if args.infer_batch_size == -1 else args.infer_batch_size}")
    
    # Initialize reward calculator and difficulty classifier
    reward_calc = RewardCalculator()
    difficulty_classifier = SampleDifficultyClassifier(max_k=max_k)
    
    # Store results for each sample
    sample_results = []
    
    # Decide single-sample or batch mode based on infer_batch_size
    if args.infer_batch_size == -1:
        # Full batch mode: pass all samples at once
        logger.info("🚀 Using full batch mode...")
        try:
            all_completions = engine.generate_n_completions_batch(
                samples=list(dataset),
                n=max_k,
                prompt_key=args.prompt_key,
                image_key=args.image_key,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            logger.error(f"Full batch generation failed: {e}")
            logger.info("⚠️ Falling back to single-sample mode...")
            all_completions = None
        
        if all_completions is not None:
            # Process batch results
            for idx, (sample, completions) in enumerate(tqdm(zip(dataset, all_completions), total=len(dataset), desc="Progress")):
                gt_rep = sample.get("output_rep", sample.get("completion", ""))
                rep_type = sample.get("output_rep_type", "can_smiles")
                
                # Compute rewards
                rewards = reward_calc.compute_group_rewards(completions, gt_rep, rep_type)
                
                # Evaluate each completion
                completion_results = []
                for comp in completions:
                    try:
                        results = compute_mol_metrics(gt_rep, comp, rep_type)
                        completion_results.append(results)
                    except Exception as e:
                        logger.error(f"Sample {idx} evaluation failed: {e}")
                        completion_results.append({
                            "is_pred_valid": False,
                            "is_inchikey_match": False,
                            "tanimoto_sim": 0.0,
                        })
                
                # Sample difficulty classification
                sample_data = {"results": completion_results}
                difficulty = difficulty_classifier.classify_sample(sample_data)
                
                sample_results.append({
                    "idx": idx,
                    "gt_rep": gt_rep,
                    "rep_type": rep_type,
                    "completions": completions,
                    "rewards": rewards,
                    "results": completion_results,
                    "difficulty": difficulty,
                })
        else:
            # Fallback to single-sample mode
            for idx, sample in enumerate(tqdm(dataset, desc="Progress")):
                gt_rep = sample.get("output_rep", sample.get("completion", ""))
                rep_type = sample.get("output_rep_type", "can_smiles")
                
                try:
                    completions = engine.generate_n_completions(
                        sample=sample,
                        n=max_k,
                        prompt_key=args.prompt_key,
                        image_key=args.image_key,
                        max_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                    )
                except Exception as e:
                    logger.error(f"Sample {idx} generation failed: {e}")
                    completions = [""] * max_k
                
                rewards = reward_calc.compute_group_rewards(completions, gt_rep, rep_type)
                
                completion_results = []
                for comp in completions:
                    try:
                        results = compute_mol_metrics(gt_rep, comp, rep_type)
                        completion_results.append(results)
                    except Exception as e:
                        logger.error(f"Sample {idx} evaluation failed: {e}")
                        completion_results.append({
                            "is_pred_valid": False,
                            "is_inchikey_match": False,
                            "tanimoto_sim": 0.0,
                        })
                
                sample_data = {"results": completion_results}
                difficulty = difficulty_classifier.classify_sample(sample_data)
                
                sample_results.append({
                    "idx": idx,
                    "gt_rep": gt_rep,
                    "rep_type": rep_type,
                    "completions": completions,
                    "rewards": rewards,
                    "results": completion_results,
                    "difficulty": difficulty,
                })
    else:
        # Chunked batch mode
        batch_size = args.infer_batch_size
        logger.info(f"🔄 Using chunked batch mode, batch_size={batch_size}...")
        
        for batch_start in tqdm(range(0, len(dataset), batch_size), desc="Batch generation"):
            batch_end = min(batch_start + batch_size, len(dataset))
            batch_samples = dataset[batch_start:batch_end]
            
            try:
                batch_completions = engine.generate_n_completions_batch(
                    samples=list(batch_samples),
                    n=max_k,
                    prompt_key=args.prompt_key,
                    image_key=args.image_key,
                    max_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            except Exception as e:
                logger.error(f"Batch {batch_start}-{batch_end} generation failed: {e}")
                batch_completions = [[""] * max_k] * len(batch_samples)
            
            # Process batch results
            for i, (sample, completions) in enumerate(zip(batch_samples, batch_completions)):
                idx = batch_start + i
                gt_rep = sample.get("output_rep", sample.get("completion", ""))
                rep_type = sample.get("output_rep_type", "can_smiles")
                
                rewards = reward_calc.compute_group_rewards(completions, gt_rep, rep_type)
                
                completion_results = []
                for comp in completions:
                    try:
                        results = compute_mol_metrics(gt_rep, comp, rep_type)
                        completion_results.append(results)
                    except Exception as e:
                        logger.error(f"Sample {idx} evaluation failed: {e}")
                        completion_results.append({
                            "is_pred_valid": False,
                            "is_inchikey_match": False,
                            "tanimoto_sim": 0.0,
                        })
                
                sample_data = {"results": completion_results}
                difficulty = difficulty_classifier.classify_sample(sample_data)
                
                sample_results.append({
                    "idx": idx,
                    "gt_rep": gt_rep,
                    "rep_type": rep_type,
                    "completions": completions,
                    "rewards": rewards,
                    "results": completion_results,
                    "difficulty": difficulty,
                })
    
    # Compute pass@k and additional diagnostic metrics
    pass_at_k_results = {}
    inchikey_match_at_k = {}
    can_smiles_match_at_k = {}
    avg_tanimoto_at_k = {}
    valid_rate_at_k = {}
    
    # GRPO specific metrics
    all_correct_at_k = {}
    all_wrong_at_k = {}
    mean_reward_at_k = {}
    std_reward_at_k = {}
    cv_reward_at_k = {}
    max_reward_at_k = {}
    min_reward_at_k = {}
    range_reward_at_k = {}
    
    # pass@k data counts
    pass_at_k_counts = {}  # Pass sample count per k (absolute)
    pass_at_k_ratios = {}  # Pass ratio per k
    
    # Incremental pass@k (first pass k value distribution)
    incremental_pass_at_k_counts = {}  # First pass absolute counts
    incremental_pass_at_k_ratios = {}  # First pass ratios
    
    # Precompute first pass k values for all samples
    first_pass_k_list = []
    for sample in sample_results:
        first_pass_k = None
        for i, r in enumerate(sample["results"]):
            if r.get("is_inchikey_match", False):
                first_pass_k = i + 1
                break
        first_pass_k_list.append(first_pass_k)
    
    for k in k_values:
        pass_count = 0
        inchikey_count = 0
        smiles_count = 0
        tanimoto_scores = []
        valid_completions = 0
        total_completions = 0
        
        # GRPO statistics
        all_correct_count = 0
        all_wrong_count = 0
        mean_rewards = []
        std_rewards = []
        cv_rewards = []
        max_rewards = []
        min_rewards = []
        range_rewards = []
        
        for sample in sample_results:
            # Take first k completions statistics
            k_results = sample["results"][:k]
            k_rewards = sample["rewards"][:k]
            
            # pass@k: at least one InChIKey match (core definition)
            has_pass = any(r.get("is_inchikey_match", False) for r in k_results)
            if has_pass:
                pass_count += 1
            
            # inchikey_match@k: at least one InChIKey match (same as pass@k)
            has_inchikey = any(r.get("is_inchikey_match", False) for r in k_results)
            if has_inchikey:
                inchikey_count += 1
            
            # can_smiles_match@k: at least one Canonical SMILES match
            has_smiles = any(r.get("is_can_smiles_match", False) for r in k_results)
            if has_smiles:
                smiles_count += 1
            
            # avg_best_tanimoto@k: best Tanimoto similarity among first k
            best_tanimoto = max(
                [r.get("tanimoto_sim", 0.0) for r in k_results],
                default=0.0
            )
            tanimoto_scores.append(best_tanimoto)
            
            # valid_rate@k: valid ratio among first k (auxiliary metric)
            k_valid_count = sum(1 for r in k_results if r.get("is_pred_valid", False))
            valid_completions += k_valid_count
            total_completions += len(k_results)
            
            # === GRPO metrics ===
            # all_correct@k: all k InChIKey matches
            if all(r.get("is_inchikey_match", False) for r in k_results):
                all_correct_count += 1
            
            # all_wrong@k: no InChIKey matches among k
            if not any(r.get("is_inchikey_match", False) for r in k_results):
                all_wrong_count += 1
            
            # Reward statistics
            k_rewards_values = k_rewards  # Already float list
            mean_r = np.mean(k_rewards_values)
            std_r = np.std(k_rewards_values)
            cv_r = std_r / (mean_r + 1e-8)
            max_r = np.max(k_rewards_values)
            min_r = np.min(k_rewards_values)
            range_r = max_r - min_r
            
            mean_rewards.append(mean_r)
            std_rewards.append(std_r)
            cv_rewards.append(cv_r)
            max_rewards.append(max_r)
            min_rewards.append(min_r)
            range_rewards.append(range_r)
        
        n_samples = len(sample_results)
        pass_at_k_results[f"pass@{k}"] = pass_count / n_samples
        pass_at_k_counts[f"pass@{k}"] = pass_count  # Pass sample count (absolute)
        pass_at_k_ratios[f"pass@{k}"] = pass_count / n_samples  # Pass ratio
        
        # Incremental pass@k: count samples with first pass k <= k and > previous k
        # For k=1: first_pass_k == 1
        # For k>1: prev_k < first_pass_k <= k
        prev_k = k_values[k_values.index(k) - 1] if k_values.index(k) > 0 else 0
        incremental_pass_count = sum(1 for fpk in first_pass_k_list if fpk is not None and prev_k < fpk <= k)
        incremental_pass_at_k_counts[f"incremental_pass@{k}"] = incremental_pass_count
        incremental_pass_at_k_ratios[f"incremental_pass@{k}"] = incremental_pass_count / n_samples
        
        inchikey_match_at_k[f"inchikey_match@{k}"] = inchikey_count / n_samples
        can_smiles_match_at_k[f"can_smiles_match@{k}"] = smiles_count / n_samples
        avg_tanimoto_at_k[f"avg_best_tanimoto@{k}"] = np.mean(tanimoto_scores)
        valid_rate_at_k[f"valid_rate@{k}"] = valid_completions / total_completions if total_completions > 0 else 0.0
        
        # GRPO metrics
        all_correct_at_k[f"all_correct@{k}"] = all_correct_count / n_samples
        all_wrong_at_k[f"all_wrong@{k}"] = all_wrong_count / n_samples
        mean_reward_at_k[f"mean_reward@{k}"] = np.mean(mean_rewards)
        std_reward_at_k[f"std_reward@{k}"] = np.mean(std_rewards)
        cv_reward_at_k[f"cv_reward@{k}"] = np.mean(cv_rewards)
        max_reward_at_k[f"max_reward@{k}"] = np.mean(max_rewards)
        min_reward_at_k[f"min_reward@{k}"] = np.mean(min_rewards)
        range_reward_at_k[f"range_reward@{k}"] = np.mean(range_rewards)
        
        logger.info(f"✅ pass@{k} = {pass_at_k_results[f'pass@{k}']:.4f} ({pass_count}/{n_samples})")
        logger.info(f"🔑 inchikey_match@{k} = {inchikey_match_at_k[f'inchikey_match@{k}']:.4f} ({inchikey_count}/{n_samples})")
        logger.info(f"🧪 can_smiles_match@{k} = {can_smiles_match_at_k[f'can_smiles_match@{k}']:.4f} ({smiles_count}/{n_samples})")
        logger.info(f"📈 avg_best_tanimoto@{k} = {avg_tanimoto_at_k[f'avg_best_tanimoto@{k}']:.4f}")
        logger.info(f"📊 valid_rate@{k} = {valid_rate_at_k[f'valid_rate@{k}']:.4f} ({valid_completions}/{total_completions})")
        logger.info(f"📈 incremental_pass@{k} = {incremental_pass_at_k_ratios[f'incremental_pass@{k}']:.4f} ({incremental_pass_count}/{n_samples})")
        logger.info(f"🎯 all_correct@{k} = {all_correct_at_k[f'all_correct@{k}']:.4f} ({all_correct_count}/{n_samples})")
        logger.info(f"❌ all_wrong@{k} = {all_wrong_at_k[f'all_wrong@{k}']:.4f} ({all_wrong_count}/{n_samples})")
        logger.info(f"💰 mean_reward@{k} = {mean_reward_at_k[f'mean_reward@{k}']:.4f}")
        logger.info(f"📉 std_reward@{k} = {std_reward_at_k[f'std_reward@{k}']:.4f}")
        logger.info(f"📊 cv_reward@{k} = {cv_reward_at_k[f'cv_reward@{k}']:.4f}")
    
    # Sample difficulty distribution statistics
    difficulty_distribution = difficulty_classifier.classify_dataset(sample_results)
    logger.info(f"📋 Sample difficulty distribution: {difficulty_distribution}")
    
    # Print pass@k data ratios
    logger.info(f"📊 pass@k ratios:")
    for k in k_values:
        count = pass_at_k_counts.get(f"pass@{k}", 0)
        ratio = pass_at_k_results.get(f"pass@{k}", 0.0)
        logger.info(f"  pass@{k}: {count}/{n_samples} = {ratio:.4f} ({ratio*100:.2f}%)")
    
    # Print incremental pass@k data ratios
    logger.info(f"📊 incremental_pass@k ratios (first pass):")
    for k in k_values:
        count = incremental_pass_at_k_counts.get(f"incremental_pass@{k}", 0)
        ratio = incremental_pass_at_k_ratios.get(f"incremental_pass@{k}", 0.0)
        logger.info(f"  incremental_pass@{k}: {count}/{n_samples} = {ratio:.4f} ({ratio*100:.2f}%)")
    impossible_count = n_samples - sum(incremental_pass_at_k_counts.values())
    logger.info(f"  impossible: {impossible_count}/{n_samples} = {impossible_count/n_samples:.4f} ({impossible_count/n_samples*100:.2f}%)")
    
    # Aggregate results
    final_results = {
        "engine_name": args.engine_name,
        "dataset_name": args.dataset_name,
        "dataset_split": args.dataset_split,
        "num_samples": len(dataset),
        "k_values": k_values,
        "pass_at_k": pass_at_k_results,
        "inchikey_match_at_k": inchikey_match_at_k,
        "can_smiles_match_at_k": can_smiles_match_at_k,
        "avg_tanimoto_at_k": avg_tanimoto_at_k,
        "valid_rate_at_k": valid_rate_at_k,
        "all_correct_at_k": all_correct_at_k,
        "all_wrong_at_k": all_wrong_at_k,
        "mean_reward_at_k": mean_reward_at_k,
        "std_reward_at_k": std_reward_at_k,
        "cv_reward_at_k": cv_reward_at_k,
        "max_reward_at_k": max_reward_at_k,
        "min_reward_at_k": min_reward_at_k,
        "range_reward_at_k": range_reward_at_k,
        "pass_at_k_counts": pass_at_k_counts,  # pass@k absolute counts
        "pass_at_k_ratios": pass_at_k_ratios,  # pass@k ratios
        "incremental_pass_at_k_counts": incremental_pass_at_k_counts,  # incremental pass absolute counts
        "incremental_pass_at_k_ratios": incremental_pass_at_k_ratios,  # incremental pass ratios
        "difficulty_distribution": difficulty_distribution,
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "sample_results": sample_results,  # Detailed results
    }
    
    return final_results


def save_difficulty_indices(results: Dict[str, Any], output_file: str):
    """Save difficulty indices to JSON file"""
    if output_file is None:
        return
    
    # Build indices
    indices = {"easy": [], "medium": [], "hard": [], "impossible": []}
    for sample in results["sample_results"]:
        difficulty = sample["difficulty"]
        indices[difficulty].append(sample["idx"])
    
    # Add metadata
    indices["metadata"] = {
        "dataset_name": results["dataset_name"],
        "dataset_split": results["dataset_split"],
        "total_samples": results["num_samples"],
        "engine_name": results["engine_name"],
        "generation_params": results["generation_params"],
    }
    
    # Save full indices
    indices_file = output_file.replace(".json", "_indices.json").replace(".jsonl", "_indices.json").replace(".csv", "_indices.json")
    if not indices_file.endswith("_indices.json"):
        indices_file = output_file + "_indices.json"
    
    os.makedirs(os.path.dirname(indices_file) if os.path.dirname(indices_file) else ".", exist_ok=True)
    with open(indices_file, "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Difficulty indices saved to: {indices_file}")
    
    # Save mixed indices (medium + hard)
    mixed_indices = indices["medium"] + indices["hard"]
    mixed_file = indices_file.replace("_indices.json", "_mixed_indices.json")
    with open(mixed_file, "w", encoding="utf-8") as f:
        json.dump(mixed_indices, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Mixed indices (medium+hard) saved to: {mixed_file}")
    
    # Print statistics
    logger.info(f"📊 Difficulty index statistics:")
    for difficulty in ["easy", "medium", "hard", "impossible"]:
        logger.info(f"  {difficulty}: {len(indices[difficulty])} samples")
    logger.info(f"  mixed (medium+hard): {len(mixed_indices)} samples")


def save_results(results: Dict[str, Any], output_file: str):
    """Save results to file"""
    if output_file is None:
        return
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    
    # 1. Save summary statistics as CSV
    csv_file = output_file.replace(".json", ".csv").replace(".jsonl", ".csv")
    if not csv_file.endswith(".csv"):
        csv_file = output_file + ".csv"
    
    import pandas as pd
    from collections import OrderedDict
    
    summary_data = []
    for k in results["k_values"]:
        row = OrderedDict([
            # pass@k and incremental count/ratio first
            ("k", k),
            ("pass_at_k_count", results["pass_at_k_counts"].get(f"pass@{k}", 0)),
            ("pass_at_k_ratio", results["pass_at_k_ratios"].get(f"pass@{k}", 0.0)),
            ("incremental_pass_count", results["incremental_pass_at_k_counts"].get(f"incremental_pass@{k}", 0)),
            ("incremental_pass_ratio", results["incremental_pass_at_k_ratios"].get(f"incremental_pass@{k}", 0.0)),
            # other metrics
            ("pass_at_k", results["pass_at_k"].get(f"pass@{k}", 0.0)),
            ("inchikey_match_at_k", results["inchikey_match_at_k"].get(f"inchikey_match@{k}", 0.0)),
            ("can_smiles_match_at_k", results["can_smiles_match_at_k"].get(f"can_smiles_match@{k}", 0.0)),
            ("avg_best_tanimoto_at_k", results["avg_tanimoto_at_k"].get(f"avg_best_tanimoto@{k}", 0.0)),
            ("valid_rate_at_k", results["valid_rate_at_k"].get(f"valid_rate@{k}", 0.0)),
            ("all_correct_at_k", results["all_correct_at_k"].get(f"all_correct@{k}", 0.0)),
            ("all_wrong_at_k", results["all_wrong_at_k"].get(f"all_wrong@{k}", 0.0)),
            ("mean_reward_at_k", results["mean_reward_at_k"].get(f"mean_reward@{k}", 0.0)),
            ("std_reward_at_k", results["std_reward_at_k"].get(f"std_reward@{k}", 0.0)),
            ("cv_reward_at_k", results["cv_reward_at_k"].get(f"cv_reward@{k}", 0.0)),
            ("max_reward_at_k", results["max_reward_at_k"].get(f"max_reward@{k}", 0.0)),
            ("min_reward_at_k", results["min_reward_at_k"].get(f"min_reward@{k}", 0.0)),
            ("range_reward_at_k", results["range_reward_at_k"].get(f"range_reward@{k}", 0.0)),
        ])
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(csv_file, index=False)
    logger.info(f"💾 Summary results saved to: {csv_file}")
    
    # Save difficulty distribution to separate CSV
    difficulty_csv = csv_file.replace(".csv", "_difficulty.csv")
    difficulty_df = pd.DataFrame([results["difficulty_distribution"]])
    difficulty_df.to_csv(difficulty_csv, index=False)
    logger.info(f"💾 Difficulty distribution saved to: {difficulty_csv}")
    
    # 2. Save raw LLM responses as JSONL
    jsonl_file = output_file.replace(".csv", ".jsonl").replace(".json", ".jsonl")
    if not jsonl_file.endswith(".jsonl"):
        jsonl_file = output_file if output_file.endswith(".jsonl") else output_file + ".jsonl"
    
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for sample in results["sample_results"]:
            record = {
                "idx": sample["idx"],
                "gt_rep": sample["gt_rep"],
                "rep_type": sample["rep_type"],
                "completions": sample["completions"],
                "rewards": sample["rewards"],
                "difficulty": sample["difficulty"],
                "results": sample["results"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"💾 Raw LLM responses saved to: {jsonl_file}")
    
    # 3. Save full results (with config and metadata) as JSON
    json_file = output_file.replace(".csv", ".json").replace(".jsonl", ".json")
    if not json_file.endswith(".json"):
        json_file = output_file + ".json"
    
    # Remove sample_results to reduce file size
    compact_results = {k: v for k, v in results.items() if k != "sample_results"}
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(compact_results, f, indent=2, ensure_ascii=False)
    logger.info(f"💾 Full results (without raw responses) saved to: {json_file}")


def main():
    parser = argparse.ArgumentParser(description="Pass@k Evaluation Script")
    parser.add_argument("--engine_name", type=str, required=True, help="VLLMEngine config name")
    parser.add_argument("--config_path", type=str, default="configs/model_inference_schema.yaml", help="Inference config file path")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--dataset_split", type=str, default="valid", help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=-1, help="Maximum samples")
    parser.add_argument("--max_k", type=int, default=16, help="Maximum k value")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum generation tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="top_p parameter")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Prompt field name in dataset")
    parser.add_argument("--image_key", type=str, default="input_rep", help="Image field name in dataset")
    parser.add_argument("--infer_batch_size", type=int, default=-1, help="Inference batch size, -1 for full batch, >0 for chunked")
    
    args = parser.parse_args()
    eval_args = EvalArguments(**vars(args))
    
    logger.info("=" * 60)
    logger.info("🚀 Pass@k Evaluation Started")
    logger.info("=" * 60)
    logger.info(f"Engine: {eval_args.engine_name}")
    logger.info(f"Dataset: {eval_args.dataset_name} ({eval_args.dataset_split})")
    logger.info(f"K values: {get_k_values(eval_args.max_k)}")
    logger.info("=" * 60)
    
    # 1. Load dataset
    logger.info(f"📚 Loading dataset: {eval_args.dataset_name}")
    dataset = load_dataset(eval_args.dataset_name, split=eval_args.dataset_split)
    
    if eval_args.max_samples != -1 and eval_args.max_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(min(eval_args.max_samples, len(dataset))))
        logger.info(f"📝 Using {len(dataset)} samples (random shuffle) for evaluation")
    else:
        logger.info(f"📝 Using all samples {len(dataset)} for evaluation...")
    
    # 2. Initialize PassAtK VLLMEngine
    logger.info(f"📦 Initializing VLLMEngine: {eval_args.engine_name}")
    engine = PassAtKVLLMEngine(eval_args.engine_name, config_path=eval_args.config_path)
    engine.build()  # Preload model
    logger.info(f"✅ Model loaded, modality: {engine.modality}")
    
    # 3. Run evaluation
    results = evaluate_pass_at_k(engine, dataset, eval_args)
    
    # 4. Save results
    save_results(results, eval_args.output_file)
    
    # 5. Save difficulty indices
    if eval_args.max_samples != -1:
        logger.warning(
            f"🔴🔴 Current max_samples={eval_args.max_samples}, difficulty indices based on partial samples may be inaccurate. Recommend using difficulty indices for GRPO training only with max_samples=None (full data)..."
        )
    save_difficulty_indices(results, eval_args.output_file)
    
    # 6. Print summary
    logger.info("=" * 60)
    logger.info("📊 Pass@k Evaluation Summary")
    logger.info("=" * 60)
    for k in results["k_values"]:
        logger.info(f"k={k}:")
        logger.info(f"  pass@{k}:              {results['pass_at_k'][f'pass@{k}']:.4f}")
        logger.info(f"  inchikey_match@{k}:    {results['inchikey_match_at_k'][f'inchikey_match@{k}']:.4f}")
        logger.info(f"  can_smiles_match@{k}:  {results['can_smiles_match_at_k'][f'can_smiles_match@{k}']:.4f}")
        logger.info(f"  avg_best_tanimoto@{k}: {results['avg_tanimoto_at_k'][f'avg_best_tanimoto@{k}']:.4f}")
        logger.info(f"  valid_rate@{k}:        {results['valid_rate_at_k'][f'valid_rate@{k}']:.4f}")
        logger.info(f"  all_correct@{k}:       {results['all_correct_at_k'][f'all_correct@{k}']:.4f}")
        logger.info(f"  all_wrong@{k}:         {results['all_wrong_at_k'][f'all_wrong@{k}']:.4f}")
        logger.info(f"  mean_reward@{k}:       {results['mean_reward_at_k'][f'mean_reward@{k}']:.4f}")
        logger.info(f"  std_reward@{k}:        {results['std_reward_at_k'][f'std_reward@{k}']:.4f}")
        logger.info(f"  cv_reward@{k}:         {results['cv_reward_at_k'][f'cv_reward@{k}']:.4f}")
        logger.info(f"  incremental_pass@{k}:   {results['incremental_pass_at_k_ratios'][f'incremental_pass@{k}']:.4f}")
    logger.info("=" * 60)
    logger.info(f"📋 Sample difficulty distribution: {results['difficulty_distribution']}")
    logger.info(f"📊 pass@k ratios: {results['pass_at_k_ratios']}")
    logger.info(f"📊 incremental_pass@k ratios: {results['incremental_pass_at_k_ratios']}")
    logger.info("=" * 60)
    logger.info("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
