# train_grpo.py
"""TRL GRPO Training Script for HelpdeskEnv.
Designed to run on Google Colab (T4 GPU) or HuggingFace Spaces.
Budget: ~$0.20-$0.50 on T4-small for 150 training steps.
Usage (Colab):
    1. Upload this file + training_data/ folder to Colab
    2. Run: !pip install --upgrade trl transformers datasets accelerate torch
    3. Run: !python train_grpo.py
Usage (local, CPU — slow but works for testing):
    python train_grpo.py --steps 10 --device cpu
"""
import argparse
import inspect
import json
import os
import re
import sys
import time
from typing import List, Dict, Any
# ─── Dependency check ────────────────────────────────────────
try:
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install trl transformers datasets accelerate torch")
    sys.exit(1)
# ─── Config ──────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_STEPS = 150
BATCH_SIZE = 4
LEARNING_RATE = 5e-6
MAX_PROMPT_LENGTH = 512
MAX_COMPLETION_LENGTH = 256
NUM_GENERATIONS = 4  # GRPO generates N completions per prompt
# Valid actions per role
VALID_ACTIONS = {
    "triage": ["triage"],
    "l1_support": ["search_kb", "apply_solution", "respond_to_customer", "escalate"],
    "l2_support": ["search_kb", "apply_fix", "respond_to_customer", "escalate"],
    "l3_support": ["search_kb", "apply_complex_fix", "respond_to_customer", "escalate", "write_kb_entry"],
}
TRIAGE_CATEGORIES = {"password_reset", "software_install", "network_issue", "hardware_failure", "data_recovery", "other"}
TRIAGE_PRIORITIES = {"low", "medium", "high", "critical"}
TRIAGE_TIERS = {"L1", "L2", "L3"}
# ─── Reward Function ────────────────────────────────────────
def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
    """Score each generated completion.
    Scoring dimensions:
    1. Valid JSON structure (0.0 or 0.3)
    2. Valid action_type for the role (0.0 or 0.2)
    3. Action value quality (0.0 - 0.3)
    4. Appropriate detail length (0.0 - 0.2)
    Total possible: 1.0
    """
    rewards = []
    for completion, prompt in zip(completions, prompts):
        try:
            score = _score_single(completion, prompt)
        except Exception:
            score = 0.0
        rewards.append(score)
    return rewards
def _score_single(completion: str, prompt: str) -> float:
    """Score a single completion against its prompt."""
    score = 0.0
    # Extract role from prompt
    role = "triage"
    for r in ["l3_support", "l2_support", "l1_support", "triage"]:
        if r in prompt.lower():
            role = r
            break
    # 1. Valid JSON (0.3 points)
    text = completion.strip()
    # Try to extract JSON from the completion
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        return 0.0
    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return 0.05  # At least attempted JSON
    score += 0.3
    # 2. Valid action_type (0.2 points)
    action_type = data.get("action_type", "")
    valid_actions = VALID_ACTIONS.get(role, [])
    if action_type in valid_actions:
        score += 0.2
    elif action_type:
        score += 0.05  # Some action attempted
    # 3. Action value quality (0.3 points)
    action_value = str(data.get("action_value", ""))
    if action_type == "triage":
        # Check triage JSON structure
        try:
            triage = json.loads(action_value) if isinstance(action_value, str) else action_value
            if isinstance(triage, dict):
                cat = str(triage.get("category", "")).lower()
                pri = str(triage.get("priority", "")).lower()
                tier = str(triage.get("tier", "")).upper()
                if cat in TRIAGE_CATEGORIES:
                    score += 0.1
                if pri in TRIAGE_PRIORITIES:
                    score += 0.1
                if tier in TRIAGE_TIERS:
                    score += 0.1
        except (json.JSONDecodeError, TypeError):
            pass
    else:
        # Non-triage: score based on value content
        words = action_value.split()
        if len(words) >= 10:
            score += 0.15
        elif len(words) >= 5:
            score += 0.1
        if len(words) >= 20:
            score += 0.1
        # Bonus for relevant keywords
        relevant_kw = ["resolved", "fix", "install", "password", "network",
                       "diagnosed", "verified", "applied", "steps"]
        hits = sum(1 for kw in relevant_kw if kw in action_value.lower())
        score += min(hits * 0.02, 0.05)
    # 4. Appropriate length (0.2 points)
    total_words = len(text.split())
    if 20 <= total_words <= 300:
        score += 0.2
    elif 10 <= total_words < 20:
        score += 0.1
    elif total_words > 300:
        score += 0.1
    return min(score, 1.0)
# ─── Dataset Loading ─────────────────────────────────────────
def load_training_data(path: str = "training_data/helpdesk_train.jsonl") -> Dataset:
    """Load JSONL training data into a HuggingFace Dataset."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line.strip())
            examples.append({"prompt": ex["prompt"]})
    return Dataset.from_list(examples)
# ─── Main Training Loop ──────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train HelpdeskEnv agent with GRPO")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/grpo_model")
    args = parser.parse_args()
    print("=" * 60)
    print("HelpdeskEnv -- GRPO Training")
    print("=" * 60)
    print(f"  Model:       {MODEL_NAME}")
    print(f"  Steps:       {args.steps}")
    print(f"  Batch size:  {BATCH_SIZE}")
    print(f"  LR:          {LEARNING_RATE}")
    print(f"  Device:      {args.device}")
    # Load dataset
    dataset = load_training_data()
    print(f"  Dataset:     {len(dataset)} prompts")
    # Load model
    print(f"\n  Loading model...")
    load_kwargs = {
        "device_map": args.device if args.device != "cpu" else None,
    }
    # Handle torch_dtype vs dtype deprecation
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **load_kwargs,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            **load_kwargs,
        )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # GRPO Config — version-safe parameter detection
    grpo_params = {
        "output_dir": args.output_dir,
        "num_train_epochs": 1,
        "max_steps": args.steps,
        "per_device_train_batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_completion_length": MAX_COMPLETION_LENGTH,
        "num_generations": NUM_GENERATIONS,
        "logging_steps": 5,
        "save_steps": 50,
        "report_to": "none",
        "log_level": "info",
        "bf16": torch.cuda.is_available(),
    }
    # Only add max_prompt_length if the installed TRL version supports it
    _config_sig = inspect.signature(GRPOConfig.__init__)
    if "max_prompt_length" in _config_sig.parameters:
        grpo_params["max_prompt_length"] = MAX_PROMPT_LENGTH
    else:
        print(f"  [INFO] TRL version does not support max_prompt_length, skipping")
    config = GRPOConfig(**grpo_params)
    # Trainer — handle tokenizer vs processing_class API change
    print(f"  Initializing GRPO trainer...")
    _trainer_sig = inspect.signature(GRPOTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "args": config,
        "train_dataset": dataset,
        "reward_funcs": [reward_fn],
    }
    if "processing_class" in _trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = GRPOTrainer(**trainer_kwargs)
    # Train
    print(f"\n{'_' * 60}")
    print(f"  Starting training ({args.steps} steps)...")
    print(f"{'_' * 60}\n")
    start_time = time.time()
    try:
        train_result = trainer.train()
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Training failed after {elapsed:.0f}s: {e}")
        # Still try to save whatever metrics we have
        _save_metrics(trainer, elapsed, args, error=str(e))
        raise
    elapsed = time.time() - start_time
    print(f"\n{'_' * 60}")
    print(f"  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'_' * 60}")
    # Save model
    try:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"  Model saved to: {args.output_dir}")
    except Exception as e:
        print(f"  [WARNING] Could not save model: {e}")
    # Save metrics
    _save_metrics(trainer, elapsed, args)
    print(f"\n{'=' * 60}")
    print("Training complete! Next: python generate_plots.py")
    print(f"{'=' * 60}")

def _save_metrics(trainer, elapsed: float, args, error: str = None):
    """Save training metrics to JSON. Captures ALL log entries."""
    # Get the full log history -- don't filter, capture everything
    full_log = []
    if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
        full_log = list(trainer.state.log_history)
    # Debug: show what keys TRL actually logged
    if full_log:
        sample_keys = set()
        for entry in full_log:
            sample_keys.update(entry.keys())
        print(f"  [DEBUG] Log history: {len(full_log)} entries")
        print(f"  [DEBUG] Available keys: {sorted(sample_keys)}")
        # Print last few entries so user can see what data looks like
        for entry in full_log[-3:]:
            print(f"  [DEBUG] Sample entry: {entry}")
    else:
        print(f"  [WARNING] No log history found in trainer.state")
        # Check if trainer has any other metric attributes
        for attr in ["_metrics", "metrics", "log_history"]:
            if hasattr(trainer, attr):
                print(f"  [DEBUG] Found trainer.{attr}")
    # Determine final loss
    final_loss = None
    try:
        if hasattr(trainer, "state") and hasattr(trainer.state, "log_history"):
            for entry in reversed(full_log):
                for key in ["loss", "train_loss", "loss/policy", "train/loss"]:
                    if key in entry:
                        final_loss = entry[key]
                        break
                if final_loss is not None:
                    break
    except Exception:
        pass
    # Build metrics dict
    metrics_data = {
        "model": MODEL_NAME,
        "steps": args.steps,
        "elapsed_seconds": round(elapsed, 1),
        "final_loss": final_loss,
        "log_history": full_log,
    }
    if error:
        metrics_data["error"] = error
    # Save to both relative and absolute paths to be safe
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "training_metrics.json")
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics_data, f, indent=2, default=str)
        print(f"  Metrics saved to: {metrics_path}")
        print(f"  File size: {os.path.getsize(metrics_path)} bytes")
    except Exception as e:
        print(f"  [ERROR] Could not save metrics file: {e}")
        # Last resort: print JSON to stdout so user can copy it
        print(f"  [FALLBACK] Dumping metrics to stdout:")
        print(json.dumps(metrics_data, indent=2, default=str))

if __name__ == "__main__":
    main()