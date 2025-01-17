# AI Image Generation Program

This repository contains an AI-based image generation program designed for a technical test. The program uses the Stable Diffusion model to generate images in various styles and enhances face quality using the GFPGAN model. Additionally, it supports the integration of custom models like `dreamshaper_8.safetensors`.

## Features

- **Style Transfer**: Generates images in specified styles (e.g., Roman, superhero, fantasy, LinkedIn, wizard).
- **Face Enhancement**: Enhances the quality of faces in the generated images using GFPGAN.
- **Gender Detection**: Detects the gender of faces in the input image and adjusts the generation prompts accordingly.
- **Configurable Parameters**: Allows customization of inference steps, guidance scale, and strength for image generation.
- **Random Seed Generation**: Generates images with random seeds for variability.
- **Optimal GPU Memory Usage**: Configures PyTorch to use all available memory on the GPU.
- **Custom Model Integration**: Supports custom models like `dreamshaper_8.safetensors`.
- **Diffusers Library**: Utilizes the Diffusers library for efficient and scalable image generation.

## Requirements

- Python 3.8+
- PIP for package installation
- CUDA-enabled GPU (for optimal performance, but CPU mode is also supported)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/danigh99/labhouse-ai-tech-test.git
    cd labhouse-ai-tech-test
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Download the custom model:**

    Download the `dreamshaper_8.safetensors` model from [Civitai](https://civitai.com/models/4384/dreamshaper) and place it in the `models` folder on the root project path.

## Usage

1. **Prepare your input image:**

    Place your input image in the `input` folder and update the `input_image_path` in the code if necessary.

2. **Run the program with the desired style:**

    ```sh
    python main.py --style roman
    python main.py --style superhero
    python main.py --style linkedin
    python main.py --style wizard
    python main.py --style fantasy
    ```

3. **Generated images:**

    The enhanced images will be saved in the `output` folder with names indicating the configuration used (steps, scale, strength, seed).

## Customization

- **Add New Styles**: Update the `prompts.json` file to add new styles with corresponding positive and negative prompts.
- **Adjust Parameters**: Modify `num_inference_steps`, `guidance_scale`, and `strength` parameters in `main.py` or `image_generator.py` to fine-tune the image generation process.

## Diffusers Library

The program leverages the Diffusers library from Hugging Face for efficient and scalable image generation. The Diffusers library provides:

- **Ease of Use**: Simplifies the process of integrating different models and schedulers.
- **Flexibility**: Allows easy customization and experimentation with different model parameters.
- **Performance**: Optimized for high performance on both GPUs and CPUs, making the image generation process faster and more efficient.

## Example

```sh
python main.py --style roman
