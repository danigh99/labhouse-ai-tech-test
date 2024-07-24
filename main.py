import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import random
import torch
from image_generator import ImageGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate images with different styles.")
    parser.add_argument('--style', type=str, required=True, help='The style of the generated image (e.g., roman, superhero, fantasy, linkedin, wizard)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for image generation')
    args = parser.parse_args()

    # Configurar PyTorch para usar toda la memoria disponible
    torch.cuda.empty_cache()  # Vaciar caché de GPU
    torch.cuda.set_per_process_memory_fraction(1.0)  # Intentar usar el 100% de la memoria GPU

    model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    model_path = 'GFPGANv1.3.pth'
    custom_model_path = 'models/dreamshaper_8.safetensors'
    prompts_path = 'prompts.json'
    input_image_path = "input/6.jpg"
    output_folder = "output"

    image_generator = ImageGenerator(model_url, model_path, custom_model_path, prompts_path)
    image_generator.download_model()
    image_generator.load_pipeline()
    image_generator.load_gfpgan()

    for steps in [50, 100, 150, 200]:
        for scale in [7, 7.5, 8]:
            for strength in [0.3]:
                for i in range(10):
                    seed = random.randint(1, 10**5) if args.seed is None else args.seed + i
                    image_generator.generate_image(input_image_path, output_folder, args.style, num_inference_steps=steps, guidance_scale=scale, strength=strength, seed=seed)

if __name__ == "__main__":
    main()
