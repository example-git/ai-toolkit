import torch
from toolkit.stable_diffusion_model import StableDiffusion, ModelConfig
from toolkit.config_modules import GenerateImageConfig
import os
import re
from diffusers import FluxImg2ImgPipeline


class FluxSampler:
    def __init__(self, model_path, lora_path=None, device='cuda', dtype='fp16', low_vram=False, quantize=False):
        self.device = torch.device(device)
        self.dtype = dtype
        self.torch_dtype = torch.float16 if dtype == 'fp16' else torch.float32
        model_config = ModelConfig(
            name_or_path=model_path,
            is_flux=True,
            device=device,
            dtype=dtype,
            low_vram=low_vram,
            quantize=quantize,
            lora_path=lora_path,  # Add lora_path to ModelConfig
            sampler='flowmatch',  # Use 'flowmatch' sampler as in training
        )
        self.sd_model = StableDiffusion(
            device=device,
            model_config=model_config,
            dtype=dtype,
        )
        self.sd_model.load_model()
        # Initialize the FluxImg2ImgPipeline
        self.pipeline = FluxImg2ImgPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)

    @torch.no_grad()
    def sample(
        self,
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=28,
        seed=0,
        init_image=None,
        strength=0.75,
    ):
        output_dir = '/workspace/output/'
        os.makedirs(output_dir, exist_ok=True)

        # Find the highest existing index
        existing_files = [f for f in os.listdir(output_dir) if f.startswith('testoutput') and f.endswith('.png')]
        indices = []
        for filename in existing_files:
            match = re.search(r'testoutput_(\d+)\.png', filename)
        if match:
            indices.append(int(match.group(1)))
        if indices:
            next_index = max(indices) + 1
        else:
            next_index = 1

        # Set the output path with incremented index
        output_path = os.path.join(output_dir, f'testoutput_{next_index}.png')

        gen_config = GenerateImageConfig(
            prompt=prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_path=output_path,  # Add a file extension
            negative_prompt='',   # Empty negative prompt for FLUX models
        )
        # Generate images using the pipeline
        image = self.pipeline(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        return image


# Example usage:
flux_sampler = FluxSampler(
    model_path='/workspace/flux-dev',
    lora_path='/workspace/output/sam-oct-flux-two/sam-oct-flux-two.safetensors',
)
images = flux_sampler.sample(prompt='An anime illustration of glowing_exampleuser0 blending modern sporty attire. Dressed in casual sportswear her eyes glow orange hinting at mal intent. Her relaxed pose contrasts with the intense gaze balancing the ordinary and extraordinary. Shes in tostky_basement sam_username_example')

