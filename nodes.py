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
                "dtype": (["float16", "bfloat16", "float32"], {"default": "bfloat16"}),
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
    RETURN_TYPES = ("IMG_EMBEDS",)

    def process(self, processor, images):
        img_embeds = processor(
            images=images,
            do_rescale=False,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        return (img_embeds,)


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
    RETURN_TYPES = ("TXT_EMBEDS",)

    def process(self, processor, text):
        txt_embeds = processor(
            text=text,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        return (txt_embeds,)


class Selector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PS_MODEL",),
                "img_embeds": ("IMG_EMBEDS",),
                "txt_embeds": ("TXT_EMBEDS",),
                "threshold": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.01}),
                "count": ("INT", {"default": 1, "min": 1, "max": 1024}),
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
        img_embeds,
        txt_embeds,
        threshold,
        count,
        images=None,
        latents=None,
        masks=None,
    ):
        with torch.no_grad():
            img_embeds.to(model.device)
            img_embeds = model.get_image_features(**img_embeds)
            img_embeds = img_embeds / torch.norm(img_embeds, dim=-1, keepdim=True)

            txt_embeds.to(model.device)
            txt_embeds = model.get_text_features(**txt_embeds)
            txt_embeds = txt_embeds / torch.norm(txt_embeds, dim=-1, keepdim=True)
            scores = model.logit_scale.exp() * (txt_embeds.float() @ img_embeds.float().T)[0]

            if scores.shape[0] == 1:
                scores = (scores - 16) / 10
                scores = scores.clamp(0, 1)
            else:
                scores = torch.softmax(scores, dim=-1)

        scores = scores.cpu().tolist()
        scores = {k: v for k, v in enumerate(scores) if v >= threshold}
        scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)[:count]
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
