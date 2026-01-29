print("\n--- Downloading SDXL Text-to-Image LoRA training script ---")
sdxl_script_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora_sdxl.py"
sdxl_script_name = "train_text_to_image_lora_sdxl.py"
!wget -q -O {sdxl_script_name} {sdxl_script_url}
print(f"Script '{sdxl_script_name}' downloaded.")