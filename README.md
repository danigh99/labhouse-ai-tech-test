# AI Image Generation Program

This repository contains an AI-based image generation program designed for a technical test. The program uses the Stable Diffusion model to generate images in various styles and enhances faces using the GFPGAN model.

## Features

- **Style Transfer**: Generates images in specified styles (e.g., Roman, superhero).
- **Face Enhancement**: Enhances the quality of faces in the generated images using GFPGAN.
- **Configurable Parameters**: Allows customization of inference steps, guidance scale, and strength for image generation.

## Requirements

- Python 3.8+
- PIP for package installation
- CUDA-enabled GPU (for optimal performance, but CPU mode is also supported)

## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/ai-image-generation.git
    cd ai-image-generation
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

4. **Download the GFPGAN model:**

    The script will automatically download the GFPGAN model if it doesn't exist locally.

## Usage

1. **Prepare your input image:**

    Place your input image in the `input` folder and update the `input_image_path` in the code if necessary.

2. **Run the program with the desired style:**

    ```sh
    python generate_avatar.py --style roman
    ```

    or

    ```sh
    python generate_avatar.py --style superhero
    ```

3. **Generated images:**

    The enhanced images will be saved in the `output` folder with names indicating the configuration used (steps, scale, strength).

## Customization

- **Add New Styles**: Update the `self.prompts` dictionary in the `ImageGenerator` class to add new styles with corresponding positive and negative prompts.
- **Adjust Parameters**: Modify `num_inference_steps`, `guidance_scale`, and `strength` parameters to fine-tune the image generation process.

## Example

```sh
python generate_avatar.py --style roman
