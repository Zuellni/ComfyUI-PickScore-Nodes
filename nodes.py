import torch
from comfy.model_management import InterruptProcessingException
from transformers import AutoModel, AutoProcessor


class Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {"default": "yuvalkirstain/PickScore_v1"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "dtype": (["float32", "bfloat16", "float16"], {"default": "bfloat16"}),
            },
        }

    CATEGORY = "Zuellni/PickScore"
    FUNCTION = "load"
    RETURN_NAMES = ("MODEL", "PROCESSOR")
    RETURN_TYPES = ("PICKSCORE_MODEL", "PICKSCORE_PROCESSOR")

    def load(self, path, device, dtype):
        dtype = torch.float32 if device == "cpu" else getattr(torch, dtype)
        model = AutoModel.from_pretrained(path, torch_dtype=dtype).eval().to(device)
        processor = AutoProcessor.from_pretrained(path, torch_dtype=dtype)

        return (model, processor)


class ImageProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processor": ("PICKSCORE_PROCESSOR",),
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "Zuellni/PickScore"
    FUNCTION = "process"
    RETURN_TYPES = ("IMAGE_EMBEDS",)

    def process(self, processor, images):
        image_embeds = processor(
            images=images,
            do_rescale=False,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        return (image_embeds,)


class TextProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processor": ("PICKSCORE_PROCESSOR",),
                "text": ("STRING", {"multiline": True}),
            },
        }

    CATEGORY = "Zuellni/PickScore"
    FUNCTION = "process"
    RETURN_TYPES = ("TEXT_EMBEDS",)

    def process(self, processor, text):
        text_embeds = processor(
            text=text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        return (text_embeds,)


class Selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PICKSCORE_MODEL",),
                "image_embeds": ("IMAGE_EMBEDS",),
                "text_embeds": ("TEXT_EMBEDS",),
                "threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "count": ("INT", {"default": 1, "min": 0, "max": 1024}),

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

    def select(self, model, image_embeds, text_embeds, threshold, count, images=None, latents=None, masks=None):
        if not count:
            raise InterruptProcessingException()

        with torch.no_grad():
            image_embeds.to(model.device)
            image_embeds = model.get_image_features(**image_embeds)
            image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)

            text_embeds.to(model.device)
            text_embeds = model.get_text_features(**text_embeds)
            text_embeds = text_embeds / torch.norm(text_embeds, dim=-1, keepdim=True)

            scores = (model.logit_scale.exp() * (text_embeds @ image_embeds.T)[0])
            scores = torch.softmax(scores, dim=-1).cpu().tolist()

        scores = {k: v for k, v in enumerate(scores)}
        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)[:count]
        scores_str = ", ".join([str(round(v, 3)) for k, v in scores])

        if images is not None:
            images = [images[v[0]] for v in scores if v[1] >= threshold]
            images = torch.stack(images) if images else None

        if latents is not None:
            latents = latents["samples"]
            latents = [latents[v[0]] for v in scores if v[1] >= threshold]
            latents = {"samples": torch.stack(latents)} if latents else None

        if masks is not None:
            masks = [masks[v[0]] for v in scores if v[1] >= threshold]
            masks = torch.stack(masks) if masks else None

        if images is None and latents is None and masks is None:
            raise InterruptProcessingException()

        return (scores_str, images, latents, masks)
