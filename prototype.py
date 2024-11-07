import torch
from diffusers import StableDiffusionPipeline
import gradio as gr
from PIL import Image, ImageDraw

model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe.to("cpu") 

style_prompts = {
    "Realistic": "",
    "Sketch": "in a pencil sketch style",
    "Watercolor": "in watercolor style",
    "Vintage": "in a vintage photo style",
}

def generate_image_with_style(prompt, style, mask_area=None):
    full_prompt = f"{prompt} {style_prompts.get(style, '')}"
    image = pipe(full_prompt).images[0]  
    
    if mask_area:
        draw = ImageDraw.Draw(image)
        draw.rectangle(mask_area, fill="black")  
    return image

prompt_suggestions = ["A futuristic cityscape at sunset", "A serene beach at sunrise", "A fantasy castle in the clouds"]

with gr.Blocks() as demo:
    gr.Markdown("##Text-to-Image Generator")
    
    prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Describe the image you want to create")
    
    style_input = gr.Dropdown(choices=list(style_prompts.keys()), label="Choose a style", value="Realistic")
    
    mask_input = gr.Textbox(label="Mask Area (x1, y1, x2, y2)", placeholder="e.g., 50,50,150,150")
    
    generate_button = gr.Button("Generate Image")
    
    output_image = gr.Image(label="Generated Image")

    def show_suggestion(suggestion):
        return gr.update(value=suggestion)

    gr.Markdown("### Try these suggestions:")
    for suggestion in prompt_suggestions:
        gr.Button(suggestion).click(show_suggestion, inputs=None, outputs=prompt_input, queue=False)
    
    def generate(prompt, style, mask):
        mask_area = None
        if mask:
            try:
                coords = list(map(int, mask.split(',')))
                if len(coords) == 4:
                    mask_area = coords  
            except ValueError:
                pass  
        
        return generate_image_with_style(prompt, style, mask_area)
    
    generate_button.click(generate, [prompt_input, style_input, mask_input], output_image)

demo.launch(share=True)