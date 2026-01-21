
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import torch
import yaml

from diffusers import AutoPipelineForText2Image, AutoencoderKL
from compel import Compel, ReturnedEmbeddingsType

@dataclass
class ModelConfig:
    repo_or_path: str
    dtype: str = "fp16"
    device: str = "cuda"
    lora_weights: Optional[str] = None
    vae_path: Optional[str] = None
    variant: Optional[str] = "auto"

@dataclass
class InferenceConfig:
    guidance_scale: float = 7.5
    num_inference_steps: int = 40
    height: int = 768
    width: int = 768
    seed: Optional[int] = None
    negative_prompt_file: Optional[str] = None

class RealVizGenerator:
    def __init__(self, cfg_path: str):
        raw = yaml.safe_load(Path(cfg_path).read_text())
        m = raw.get("model", {})
        i = raw.get("inference", {})
        self.mcfg = ModelConfig(**m)
        self.icfg = InferenceConfig(**i)

        dtype = torch.float16 if self.mcfg.dtype.lower() == "fp16" else torch.float32

        # --- Load SDXL pipeline (RealVis XL diffusers folder or HF repo) ---
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.mcfg.repo_or_path,
            torch_dtype=dtype,
            variant=self.mcfg.variant,
            safety_checker=None,
        )

        # --- Replace VAE with LLaVA 1.5 (if provided) ---
        if self.mcfg.vae_path and Path(self.mcfg.vae_path).exists():
            vae = AutoencoderKL.from_pretrained(self.mcfg.vae_path, torch_dtype=dtype)
            self.pipe.vae = vae

        # --- Load LoRA (if any) ---
        if self.mcfg.lora_weights:
            self.pipe.load_lora_weights(self.mcfg.lora_weights)
            try:
                self.pipe.fuse_lora()
            except Exception:
                pass

        device = (
            self.mcfg.device
            if torch.cuda.is_available() and self.mcfg.device == "cuda"
            else "cpu"
        )
        self.pipe.to(device)

        # --- Setup Compel for SDXL (dual encoders) ---
        if all(
            hasattr(self.pipe, a)
            for a in ["tokenizer", "text_encoder", "tokenizer_2", "text_encoder_2"]
        ):
            self.is_sdxl = True
            self.compel = Compel(
                tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
                text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.SDXL,
                truncate_long_prompts=False,   # let Compel chunk and concat
            )
        else:
            # Fallback (shouldn’t happen for RealVis XL)
            self.is_sdxl = False
            self.compel = None

        # --- Load negative prompt text (string); we’ll convert via Compel in generate() ---
        self.negative_prompt_text = None
        if self.icfg.negative_prompt_file and Path(self.icfg.negative_prompt_file).exists():
            self.negative_prompt_text = Path(self.icfg.negative_prompt_file).read_text().strip()

    def _embed_sdxl(
        self, prompt_text: str, negative_text: Optional[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns: (prompt_embeds, pooled_prompt_embeds, neg_embeds, neg_pooled_embeds)
        """
        prompt_embeds, pooled_prompt_embeds = self.compel.build_conditioning_tensor(prompt_text)
        prompt_embeds = prompt_embeds.to(self.pipe.device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(self.pipe.device)

        neg_embeds = neg_pooled = None
        if negative_text:
            neg_embeds, neg_pooled = self.compel.build_conditioning_tensor(negative_text)
            neg_embeds = neg_embeds.to(self.pipe.device)
            neg_pooled = neg_pooled.to(self.pipe.device)

        return prompt_embeds, pooled_prompt_embeds, neg_embeds, neg_pooled

    @torch.inference_mode()
    def generate(self, prompt: str):
        # Optional deterministic seed
        generator = None
        if self.icfg.seed is not None:
            generator = torch.Generator(device=self.pipe.device).manual_seed(int(self.icfg.seed))

        if self.is_sdxl and self.compel is not None:
            prompt_embeds, pooled_prompt_embeds, neg_embeds, neg_pooled = self._embed_sdxl(
                prompt, self.negative_prompt_text
            )
            out = self.pipe(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                guidance_scale=self.icfg.guidance_scale,
                num_inference_steps=self.icfg.num_inference_steps,
                height=self.icfg.height,
                width=self.icfg.width,
                generator=generator,
            )
        else:
            # Fallback: pass strings (shouldn’t be used for RealVis XL)
            out = self.pipe(
                prompt=prompt,
                negative_prompt=self.negative_prompt_text,
                guidance_scale=self.icfg.guidance_scale,
                num_inference_steps=self.icfg.num_inference_steps,
                height=self.icfg.height,
                width=self.icfg.width,
                generator=generator,
            )

        if hasattr(out, "images"):
            return out.images[0]
        return None
