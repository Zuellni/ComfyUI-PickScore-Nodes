# ComfyUI PickScore Nodes
Image scoring nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) using [PickScore](https://github.com/yuvalkirstain/PickScore) with a batch of images to predict which ones fit a given prompt the best.

## Installation
Make sure your ComfyUI is up to date and clone the repository to `custom_nodes`:
```
git clone https://github.com/Zuellni/ComfyUI-PickScore-Nodes
```

## Nodes
Name | Description
:--- | :---
Loader | Loads scoring models from [Hugging Face](https://huggingface.co) or a given directory. The [default](https://huggingface.co/yuvalkirstain/PickScore_v1) one is around 4GB.
Processor | Takes images/text and converts them to embeddings.
Selector | Selects up to `count` best images/latents/masks. Interrupts generation if the `threshold` isn't met.

## Workflow
The image below can be opened in ComfyUI.

![workflow](https://github.com/Zuellni/ComfyUI-PickScore-Nodes/assets/123005779/769c070d-842b-4864-b9ea-2566dbeafde0)
