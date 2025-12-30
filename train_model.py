import subprocess

# --- 5. Training Parameters (Aligned with Report) ---
print("\n--- Defining LoRA Training Parameters ---")
pretrained_model = "runwayml/stable-diffusion-v1-5"
output_dir = "lora_face_model"


# --- LoRA Hyperparameters (Report Aligned) ---
training_config = {
    "model_id": "runwayml/stable-diffusion-v1-5", #
    "dataset_path": "lora_dataset",      # Local dataset path
    "output_name": "lora_face_model",
    "resolution": 512,                            #
    "batch_size": 1,                              # Optimized for memory
    "gradient_accumulation": 4,                   # Effective batch of 4
    "learning_rate": 1e-4,                        # Typical for LoRA [cite: 434]
    "max_steps": 1000,                            # [cite: 220]
    "precision": "fp16",                          # Essential for 8GB VRAM
    "checkpoint_freq": 500,
    "seed": 42
}



# --- Build the Acceleration Command ---
accelerate_command = (
    f"accelerate launch train_text_to_image_lora.py "
    f"--pretrained_model_name_or_path='{training_config['model_id']}' "
    f"--train_data_dir='{training_config['dataset_path']}' "
    f"--caption_column='text' "
    f"--resolution={training_config['resolution']} "
    f"--random_flip "
    f"--train_batch_size={training_config['batch_size']} "
    f"--gradient_accumulation_steps={training_config['gradient_accumulation']} "
    f"--gradient_checkpointing "           # Critical for 8GB VRAM
    f"--use_8bit_adam "                    # Required for memory efficiency
    f"--max_train_steps={training_config['max_steps']} "
    f"--learning_rate={training_config['learning_rate']} "
    f"--lr_scheduler='constant' "
    f"--lr_warmup_steps=0 "
    f"--seed={training_config['seed']} "
    f"--output_dir='{training_config['output_name']}' "
    f"--mixed_precision='{training_config['precision']}' "
    f"--enable_xformers_memory_efficient_attention "
    f"--checkpointing_steps={training_config['checkpoint_freq']} "
)

print(f"--- Launching Training: {training_config['model_id']} ---")
result = subprocess.run(accelerate_command, shell=True, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:")
    print(result.stderr)
    raise subprocess.CalledProcessError(result.returncode, accelerate_command)
