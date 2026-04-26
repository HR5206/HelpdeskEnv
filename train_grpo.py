import json, os, re, time, inspect
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOTrainer, GRPOConfig

# 1. Dataset
PROMPTS = []
for role, tid, subj, body in [
    ("triage","t1","Password expired","User locked out"),
    ("triage","t2","Install VS Code","Software request"),
    ("triage","t3","Network down floor 3","Switch failure"),
    ("l1_support","t1","Reset password","Active Directory reset"),
    ("l2_support","t2","Install IDE","Download and configure"),
    ("l3_support","t3","Switch crash","Failover and firmware"),
    ("l3_support","t4","DB corruption","Restore from backup"),
]:
    for i in range(12):
        PROMPTS.append({"prompt": f"You are a {role} agent.\nTicket: {tid}\nSubject: {subj}\nBody: {body}\nRespond with JSON."})
dataset = Dataset.from_list(PROMPTS)

# 2. Reward Function
def reward_fn(completions, prompts, **kwargs):
    rewards = []
    for c, p in zip(completions, prompts):
        s = 0.0
        m = re.search(r'\{.*\}', c.strip(), re.DOTALL)
        if not m: rewards.append(0.0); continue
        try: data = json.loads(m.group()); s += 0.4
        except: rewards.append(0.05); continue
        if data.get("action_type"): s += 0.3
        if len(str(data.get("action_value","")).split()) >= 3: s += 0.3
        rewards.append(min(s, 1.0))
    return rewards

# 3. Load Model in fp16
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16, device_map="auto")
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# *** THE FIX: Apply LoRA so gradients calculate in safe FP32 ***
peft_config = LoraConfig(r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, peft_config)
print("LoRA adapters applied!")

# 4. FAST Configuration
config_params = dict(
    output_dir="results/grpo_model",
    max_steps=50,                  
    per_device_train_batch_size=4,
    learning_rate=5e-5,            # slightly higher LR for LoRA
    max_completion_length=128,     
    num_generations=2,             
    logging_steps=5,
    save_steps=25,
    report_to="none",              
    fp16=True,                     
)

sig = inspect.signature(GRPOConfig.__init__)
if "max_prompt_length" in sig.parameters:
    config_params["max_prompt_length"] = 256
config = GRPOConfig(**config_params)

# 5. Trainer Setup
tsig = inspect.signature(GRPOTrainer.__init__)
trainer_kwargs = dict(model=model, args=config, train_dataset=dataset, reward_funcs=[reward_fn])
if "processing_class" in tsig.parameters:
    trainer_kwargs["processing_class"] = tok
else:
    trainer_kwargs["tokenizer"] = tok
trainer = GRPOTrainer(**trainer_kwargs)

# 6. Train
print(f"\nStarting GRPO training...")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"\nDone in {elapsed:.0f}s ({elapsed/60:.1f} min)")

# 7. Accurately Save Metrics
os.makedirs("results", exist_ok=True)
full_log = list(trainer.state.log_history) if hasattr(trainer.state, "log_history") else []

metrics_data = {
    "model": MODEL,
    "steps": 50,
    "elapsed_seconds": round(elapsed, 1),
    "log_history": full_log,
}

metrics_path = "results/training_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_data, f, indent=2, default=str)

print(f"Metrics saved to {metrics_path} ({os.path.getsize(metrics_path)} bytes)")

# 8. Auto-Download
from google.colab import files
files.download(metrics_path)
