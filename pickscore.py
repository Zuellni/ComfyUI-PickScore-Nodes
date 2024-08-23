import torch
from transformers import CLIPModel, CLIPProcessor

from comfy.model_management import (
    InterruptProcessingException,
    get_torch_device,
    soft_empty_cache,
)

_CATEGORY = "zuellni/pickscore"
_MAPPING = "ZuellniPickScore"


class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "yuvalkirstain/PickScore_v1"}),
            },
        }

    CATEGORY = _CATEGORY
    FUNCTION = "setup"
    RETURN_NAMES = ("MODEL",)
    RETURN_TYPES = ("PS_MODEL",)

    def setup(self, path):
        self.device = get_torch_device()

        self.dtype = (
            torch.float32 if self.device == torch.device("cpu") else torch.float16
        )

        self.model = CLIPModel.from_pretrained(path, torch_dtype=self.dtype).eval()
        self.processor = CLIPProcessor.from_pretrained(path)

        return (self,)

    def load(self):
        self.pipeline.to(self.device)

    def offload(self):
        self.pipeline.cpu()
        soft_empty_cache()


class Processor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "images": ("IMAGE",),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    CATEGORY = _CATEGORY
    FUNCTION = "process"
    RETURN_NAMES = ("INPUTS",)
    RETURN_TYPES = ("PS_INPUTS",)

    def process(self, model, images, text):
        image_inputs = model.processor(
            images=images,
            do_rescale=False,
            return_tensors="pt",
        ).to(model.device)

        text_inputs = model.processor(
            text=text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(model.device)

        return ((image_inputs, text_inputs),)


class Selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "inputs": ("PS_INPUTS",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "limit": ("INT", {"default": 1, "min": 1, "max": 1000}),
            },
            "optional": {
                "images": ("IMAGE",),
                "latents": ("LATENT",),
                "masks": ("MASK",),
            },
        }

    CATEGORY = _CATEGORY
    FUNCTION = "select"
    RETURN_NAMES = ("SCORES", "IMAGES", "LATENTS", "MASKS")
    RETURN_TYPES = ("STRING", "IMAGE", "LATENT", "MASK")

    def select(
        self,
        model,
        inputs,
        threshold,
        limit,
        images=None,
        latents=None,
        masks=None,
    ):
        image_inputs, text_inputs = inputs
        model.load()

        with torch.inference_mode():
            image_embeds = model.pipeline.get_image_features(**image_inputs)
            image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)

            text_embeds = model.pipeline.get_text_features(**text_inputs)
            text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)

            scores = (text_embeds.float() @ image_embeds.float().T)[0]

            if scores.shape[0] > 1:
                scores = model.pipeline.logit_scale.exp() * scores
                scores = torch.softmax(scores, dim=-1)

        model.offload()
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
    f"{_MAPPING}Loader": Loader,
    f"{_MAPPING}Processor": Processor,
    f"{_MAPPING}Selector": Selector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    f"{_MAPPING}Loader": "Loader",
    f"{_MAPPING}Processor": "Processor",
    f"{_MAPPING}Selector": "Selector",
}
