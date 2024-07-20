import os
import requests
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from gfpgan import GFPGANer
import numpy as np
import argparse
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

class ImageGenerator:
    def __init__(self, model_url, model_path, pipeline_name, device='cuda'):
        self.model_url = model_url
        self.model_path = model_path
        self.pipeline_name = pipeline_name
        self.device = device
        self.pipe = None
        self.gfpgan = None
        self.prompts = {
            "roman": {
                "positive": (
                    "Cinematic portrait of a {gender} with shoulder-length hair, dressed in elegant ancient Roman attire, "
                    "wearing a laurel crown. Surrounded by a large crowd of ancient Roman people in traditional clothing. "
                    "Grand Roman architecture with marble columns in the background. Bustling Roman forum with classical sculptures and a coliseum. "
                    "High detail, dramatic lighting, realistic textures."
                ),
                "negative": (
                    "plain, empty, white background, malformed, extra limbs, poorly drawn anatomy, badly drawn, extra legs, low resolution, "
                    "blurry, Watermark, Text, censored, deformed, bad anatomy, disfigured, poorly drawn face, "
                    "mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, "
                    "disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, "
                    "cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, "
                    "disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, "
                    "abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers, drawing, painting, "
                    "crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, statue, marble, stone, sculpture"
                )
            },
            "superhero": {
                "positive": (
                    "Epic portrait of a {gender} superhero in a high-tech suit with glowing accents, standing heroically on a skyscraper rooftop, flowing cape, muscular build, determined expression. Futuristic city skyline with neon lights, dramatic clouds, and bright moonlight. Flying cars and drones in the sky. High detail, dynamic lighting, vibrant colors, reflections on the suit. Smoke and sparks."
                ),
                "negative": (
                    "plain, empty, white background, malformed, extra limbs, poorly drawn anatomy, badly drawn, extra legs, low resolution, "
                    "blurry, Watermark, Text, censored, deformed, bad anatomy, disfigured, poorly drawn face, "
                    "mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, "
                    "disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, "
                    "cropped, worst quality, low quality, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, "
                    "disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, "
                    "abnormal eye proportion, abnormal hands, abnormal legs, abnormal feet, abnormal fingers, drawing, painting, "
                    "crayon, sketch, graphite, impressionist, noisy, blurry, soft, deformed, ugly, statue, marble, stone, sculpture"
                )
            }
        }

    def download_model(self):
        if not os.path.exists(self.model_path):
            response = requests.get(self.model_url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            progress = 0
            with open(self.model_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress += len(data)
                    file.write(data)
                    print(f'Downloading... {progress / total_size * 100:.2f}%', end='\r')
            print(f'\nModel downloaded to {self.model_path}')
    
    def load_pipeline(self):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(self.pipeline_name)
        self.pipe.to(self.device)
    
    def load_gfpgan(self):
        self.gfpgan = GFPGANer(model_path=self.model_path, upscale=2)
    
    def detect_gender(self, image_path):
        app = FaceAnalysis()
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
        
        faces = app.get(img_np)
        if not faces:
            return 'person'  # Default to 'person' if no face is detected
        
        gender = 'male' if faces[0].gender == 1 else 'female'
        
        return gender

    def generate_image(self, input_image_path, output_folder, style, num_inference_steps=100, guidance_scale=7.5, strength=0.75):
        input_image = Image.open(input_image_path).convert("RGB")
        input_image = input_image.resize((512, 512), Image.LANCZOS)

        if style not in self.prompts:
            print(f"Style '{style}' not found. Please choose a valid style.")
            return

        gender = self.detect_gender(input_image_path)
        positive_prompt = self.prompts[style]["positive"].format(gender=gender)
        negative_prompt = self.prompts[style]["negative"]

        output = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=512,
            width=512,
            hires_fix=True,
            hires_upscale=2,
            hires_denoise=0.7
        )

        image_np = np.array(output.images[0])
        _, _, enhanced_image = self.gfpgan.enhance(image_np, has_aligned=False, only_center_face=False, paste_back=True)
        enhanced_image = Image.fromarray(enhanced_image)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        enhanced_image_path = f"{output_folder}/enhanced_output_image_steps_{num_inference_steps}_scale_{guidance_scale}_strength_{strength}.png"
        enhanced_image.save(enhanced_image_path)
        print(f"Enhanced image saved to {enhanced_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate images with different styles.")
    parser.add_argument('--style', type=str, required=True, help='The style of the generated image (e.g., roman, superhero)')
    args = parser.parse_args()

    model_url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
    model_path = 'GFPGANv1.3.pth'
    pipeline_name = "CompVis/stable-diffusion-v1-4"
    input_image_path = "input/6.jpg"
    output_folder = "output"

    image_generator = ImageGenerator(model_url, model_path, pipeline_name)
    image_generator.download_model()
    image_generator.load_pipeline()
    image_generator.load_gfpgan()

    for steps in [50, 100, 150, 200]:
        for scale in [7, 7.5, 8]:
            for strength in [0.75]:
                image_generator.generate_image(input_image_path, output_folder, args.style, num_inference_steps=steps, guidance_scale=scale, strength=strength)

if __name__ == "__main__":
    main()