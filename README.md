# ComfyUI PickScore Nodes
Image scoring nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) using [PickScore](https://github.com/yuvalkirstain/PickScore).<br>
Used with a batch of images to check which ones fit a given prompt the best.

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
Selector | Selects `count` best images/latents/masks. Interrupts generation if the `threshold` isn't met.
