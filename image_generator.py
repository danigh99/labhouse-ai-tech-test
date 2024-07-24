import os
import json
import requests
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from gfpgan import GFPGANer
import numpy as np
import insightface
from insightface.app import FaceAnalysis

class ImageGenerator:
    def __init__(self, model_url, model_path, custom_model_path, prompts_path, device='cuda'):
        self.model_url = model_url
        self.model_path = model_path
        self.custom_model_path = custom_model_path
        self.prompts_path = prompts_path
        self.device = device
        self.pipe = None
        self.gfpgan = None
        self.prompts = self.load_prompts()

    def load_prompts(self):
        with open(self.prompts_path, 'r') as file:
            return json.load(file)

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
        scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085, 
            beta_end=0.012, 
            beta_schedule="scaled_linear", 
            steps_offset=1
        )
        self.pipe = StableDiffusionImg2ImgPipeline.from_single_file(self.custom_model_path, scheduler=scheduler)
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

    def generate_image(self, input_image_path, output_folder, style, num_inference_steps=50, guidance_scale=8, strength=0.75, seed=None):
        input_image = Image.open(input_image_path).convert("RGB")
        input_image = input_image.resize((512, 512), Image.LANCZOS)

        if style not in self.prompts:
            print(f"Style '{style}' not found. Please choose a valid style.")
            return

        gender = self.detect_gender(input_image_path)
        positive_prompt = self.prompts[style]["positive"].format(gender=gender)
        negative_prompt = self.prompts[style]["negative"]

        if seed is not None:
            generator = torch.manual_seed(seed)
        else:
            generator = None

        output = self.pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=input_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        )

        image_np = np.array(output.images[0])
        _, _, enhanced_image = self.gfpgan.enhance(image_np, has_aligned=False, only_center_face=False, paste_back=True)
        enhanced_image = Image.fromarray(enhanced_image)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        enhanced_image_path = f"{output_folder}/enhanced_output_image_steps_{num_inference_steps}_scale_{guidance_scale}_strength_{strength}_seed_{seed}.png"
        enhanced_image.save(enhanced_image_path)
        print(f"Enhanced image saved to {enhanced_image_path}")
