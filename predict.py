# Prediction interface for Cog

from cog import BasePredictor, Input, Path
import os
import time
import torch
import shutil
import numpy as np
from PIL import Image
from diffusers.utils import load_image
from carvekit.web.schemas.config import MLConfig
from carvekit.web.utils.init_utils import init_interface
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline
)

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REFINER = "stabilityai/stable-diffusion-xl-refiner-1.0"
CONTROL_MODEL = "diffusers/controlnet-depth-sdxl-1.0"
RM_CACHE = "rm-cache/"
SDXL_CACHE = "sdxl-cache/"
REFINER_CACHE = "refiner-cache"
CONTROL_CACHE = "control-cache"
FEATURE_CACHE = "feature-cache"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)
    
SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        # Copy over Background removal weights
        os.makedirs("/root/.cache/carvekit/checkpoints/tracer_b7/")
        os.makedirs("/root/.cache/carvekit/checkpoints/fba/")
        os.system("cp "+RM_CACHE+"/tracer_b7.pth /root/.cache/carvekit/checkpoints/tracer_b7/")
        os.system("cp "+RM_CACHE+"/fba_matting.pth /root/.cache/carvekit/checkpoints/fba/")
        # Load the interface
        self.interface = init_interface(
            MLConfig(
                segmentation_network="tracer_b7",
                preprocessing_method="none",
                postprocessing_method="fba",
                seg_mask_size=640,
                trimap_dilation=30,
                trimap_erosion=5,
                device='cuda'
            )
        )
        print("Loading depth feature extractor")
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(FEATURE_CACHE).to("cuda")
        self.feature_extractor = DPTFeatureExtractor.from_pretrained(FEATURE_CACHE)
        print("Loading controlnet depth model")
        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE,
            use_safetensors=True,
            torch_dtype=torch.float16
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            SDXL_CACHE,
            controlnet=controlnet,
            use_safetensors=True,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.control_pipe = pipe.to("cuda")
        print("Loading refiner pipeline...")
        self.refiner = DiffusionPipeline.from_pretrained(
            REFINER_CACHE,
            text_encoder_2=self.control_pipe.text_encoder_2,
            vae=self.control_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")
        t2 = time.time()
        print("Setup took: ", t2 - t1)


    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def get_depth_map(self, image):
        image = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
        with torch.no_grad(), torch.autocast("cuda"):
            depth_map = self.depth_estimator(image).predicted_depth

        height, width = image.shape[2], image.shape[3]
        depth_map = torch.nn.functional.interpolate(
            depth_map.unsqueeze(1),
            size=(height, width),
            mode="bicubic",
            align_corners=True,
        )
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        image = torch.cat([depth_map] * 3, dim=1)
        image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
        image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
        return image


    def predict(
        self,
        image: Path = Input(description="Remove background from this image"),
        product_fill: str = Input(
            description="What percentage of the image width to fill with product",
            choices=["Original", "80", "70", "60", "50", "40", "30", "20"],
            default='Original'
        ),
        prompt: str = Input(
            description="Describe the new setting for your product",
        ),
        negative_prompt: str = Input(
            description="Describe what you do not want in your setting",
            default="low quality, out of frame, illustration, 3d, sepia, painting, cartoons, sketch, watermark, text, Logo, advertisement"
        ),
        img_size: str = Input(
            description="Possible SDXL image sizes",
            choices=['512, 2048', '512, 1984', '512, 1920', '512, 1856',
            '576, 1792', '576, 1728', '576, 1664', '640, 1600',
            '640, 1536', '704, 1472', '704, 1408', '704, 1344',
            '768, 1344', '768, 1280', '832, 1216', '832, 1152',
            '896, 1152', '896, 1088', '960, 1088', '960, 1024',
            '1024, 1024', '1024, 960', '1088, 960', '1088, 896',
            '1152, 896', '1152, 832', '1216, 832', '1280, 768',
            '1344, 768', '1408, 704', '1472, 704', '1536, 640',
            '1600, 640', '1664, 576', '1728, 576', '1792, 576',
            '1856, 512', '1920, 512', '1984, 512', '2048, 512'],
            default='1024, 1024'
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Inference Steps",
            default=40
        ),
        guidance_scale: float = Input(
            description="Guidance Scale",
            default=7.5
        ),
        condition_scale: float = Input(
            description="controlnet conditioning scale for generalization",
            default=0.9,
            ge=0.3,
            le=0.9,
        ),
        num_refine_steps: int = Input(
            description="Number of steps to refine",
            default=10,
            ge=0,
            le=40
        ),
        apply_img: bool = Input(
            description="Applies the original product image to the final result",
            default=True
        ),
        seed: int = Input(
            description="Empty or 0 for a random image",
            default=None
        )
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed is None) or (seed == 0):
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        #1 - Remove bg from input image
        input_image = Image.open(image)
        input_width, input_height = input_image.size
        new_width = 2*input_width
        new_height = 2*input_height
        # Double up image to help remove bg
        input_image = input_image.resize((new_width, new_height))
        processed_bg = self.interface([input_image])[0]
        # Restore to original size
        processed_bg = processed_bg.resize((input_width, input_height))
        save_path = "1-out.png"
        processed_bg.save(save_path)
        
        # 2 - get input image size and scale factor to get to desired result img_size
        # ex: 80% of 1024x1024 is 800x800
        prod_width, prod_height = processed_bg.size
        print("Product img W:"+ str(prod_width) + ", H:" + str(prod_height))
        # convert product_fill to actual scale factor
        # ex product_fill=80% => 0.8
        scale_factor = 1.0
        if(product_fill != "Original"):
            scale_factor = float(product_fill) / 100.0
        print("Scale factor: " + str(scale_factor))
        # from img_size, split into width and height and multiply by scale factor
        # ex '512, 1024' => width=512, height=1024
        final_img_width, final_img_height = img_size.split(',')
        final_img_width = int(final_img_width)
        final_img_height = int(final_img_height)
        print("Final img W: " + str(final_img_width) + ", H:" + str(final_img_height))
        # How much final image width will be
        new_max_prod_width = int(scale_factor * final_img_width)
        # print("New product width: " + str(new_max_prod_width))
        # Get new max prod height
        aspect_ratio = float(prod_width / prod_height)
        new_max_prod_height = int(new_max_prod_width / aspect_ratio)
        # print("New product height: " + str(new_max_prod_height))
        prod_img = Image.open(save_path).convert('RGBA')
        if product_fill != "Original":
            prod_scaled = prod_img.resize((new_max_prod_width-10, new_max_prod_height-10))
        else:
            prod_scaled = prod_img

        save_path2 = "/tmp/2-out.png"
        prod_scaled.save(save_path2)


        # 3 - create a blank image of desired size with product in middle
        empty_image = Image.new('RGB', (final_img_width, final_img_height), 'white')
        # Smaller product image
        sm = 0.98
        smaller_prod = prod_scaled.resize((int(prod_scaled.width*sm), int(prod_scaled.height*sm)))
        x_offset = (final_img_width - smaller_prod.width) // 2
        y_offset = (final_img_height - smaller_prod.height) // 2
        empty_image.paste(smaller_prod, (x_offset, y_offset), smaller_prod)
        save_path3 = "3-out.png"
        empty_image.save(save_path3)

        # 4 - ControlNet depth on image
        self.control_pipe.scheduler = SCHEDULERS[scheduler].from_config(self.control_pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)
        # Depth image
        depth_image = self.get_depth_map(empty_image)
        save_path4 = "4-out.png"
        depth_image.save(save_path4)

        # 5 - ControlNet
        output = self.control_pipe(
            prompt,
            negative_prompt=negative_prompt,
            image=depth_image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=condition_scale,
            width=final_img_width,
            height=final_img_height,
            guidance_scale=guidance_scale,
            generator=generator
        )
        final_img = output.images[0]
        output_path = f"5-out.png"
        final_img.save(output_path)

        # 6 - Place product onto control img
        x_offset = (final_img_width - prod_scaled.width) // 2
        y_offset = (final_img_height - prod_scaled.height) // 2
        # larger product image
        # lg = 1.05
        # larger_prod = prod_scaled.resize((int(prod_scaled.width*lg), int(prod_scaled.height*lg)))
        final_img.paste(prod_scaled, (x_offset, y_offset), prod_scaled)
        # Original product onto control img
        output_path = f"6-out.png"
        final_img.save(output_path)

        # Return if no refinement needed
        if (num_refine_steps is None) or (num_inference_steps==0):
            return Path(output_path)
        
        # 7 - Refine final image
        common_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_refine_steps,
        }
        refiner_kwargs = {
            "image": final_img,
        }
        output = self.refiner(**common_args, **refiner_kwargs)
        output_path = f"7-out.png"
        final_blended = output.images[0]
        final_blended.save(output_path)

        # Check if user wants to paste on initial product image
        if apply_img:
            final_blended.paste(prod_scaled, (x_offset, y_offset), prod_scaled)
            output_path = f"8-out.png"
            final_blended.save(output_path)

        return Path(output_path)