import torch
from comfy.model_management import InterruptProcessingException
from transformers import AutoModel, AutoProcessor


class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "yuvalkirstain/PickScore_v1"}),
                "device": (("cuda", "cpu"),),
                "dtype": (("float16", "bfloat16", "float32"),),
            },
        }

    CATEGORY = "Zuellni/PickScore"
    FUNCTION = "load"
    RETURN_NAMES = ("MODEL", "PROCESSOR")
    RETURN_TYPES = ("PS_MODEL", "PS_PROCESSOR")

    def load(self, path, device, dtype):
        dtype = torch.float32 if device == "cpu" else getattr(torch, dtype)
        model = AutoModel.from_pretrained(path, torch_dtype=dtype).eval().to(device)
        processor = AutoProcessor.from_pretrained(path)

        return (model, processor)


class ImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processor": ("PS_PROCESSOR",),
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "Zuellni/PickScore"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE_INPUTS",)

    def process(self, processor, images):
        return (
            processor(
                images=images,
                do_rescale=False,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )["pixel_values"],
        )


class TextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processor": ("PS_PROCESSOR",),
                "text": ("STRING", {"multiline": True}),
            },
        }

    CATEGORY = "Zuellni/PickScore"
    FUNCTION = "process"
    RETURN_TYPES = ("TEXT_INPUTS",)

    def process(self, processor, text):
        return (
            processor(
                text=text,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            )["input_ids"],
        )


class Selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "image_inputs": ("IMAGE_INPUTS",),
                "text_inputs": ("TEXT_INPUTS",),
                "threshold": ("FLOAT", {"max": 1, "step": 0.001}),
                "limit": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "masks": ("MASK",),
            },
        }

    CATEGORY = "Zuellni/PickScore"
    FUNCTION = "select"
    RETURN_NAMES = ("SCORES", "IMAGES", "LATENTS", "MASKS")
    RETURN_TYPES = ("STRING", "IMAGE", "LATENT", "MASK")

    def select(
        self,
        model,
        image_inputs,
        text_inputs,
        threshold,
        limit,
        images=None,
        latents=None,
        masks=None,
    ):
        with torch.inference_mode():
            image_inputs = image_inputs.to(model.device, dtype=model.dtype)
            image_embeds = model.get_image_features(image_inputs)
            image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)

            text_inputs = text_inputs.to(model.device)
            text_embeds = model.get_text_features(text_inputs)
            text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)

            scores = (text_embeds.float() @ image_embeds.float().T)[0]

            if scores.shape[0] > 1:
                scores = model.logit_scale.exp() * scores
                scores = torch.softmax(scores, dim=-1)

        scores = scores.cpu().tolist()
        scores = {k: v for k, v in enumerate(scores) if v >= threshold}
        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)[:limit]
        scores_str = ", ".join([str(round(v, 3)) for k, v in scores])

        if images is not None:
            images = [images[v[0]] for v in scores]
            images = torch.stack(images) if images else None

        if latents is not None:
            latents = latents["samples"]
            latents = [latents[v[0]] for v in scores]
            latents = {"samples": torch.stack(latents)} if latents else None

        if masks is not None:
            masks = [masks[v[0]] for v in scores]
            masks = torch.stack(masks) if masks else None

        if images is None and latents is None and masks is None:
            raise InterruptProcessingException()

        return (scores_str, images, latents, masks)


NODE_CLASS_MAPPINGS = {
    "ZuellniPickScoreLoader": Loader,
    "ZuellniPickScoreImageProcessor": ImageProcessor,
    "ZuellniPickScoreTextProcessor": TextProcessor,
    "ZuellniPickScoreSelector": Selector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZuellniPickScoreLoader": "Loader",
    "ZuellniPickScoreImageProcessor": "Image Processor",
    "ZuellniPickScoreTextProcessor": "Text Processor",
    "ZuellniPickScoreSelector": "Selector",
}
