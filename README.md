# ComfyUI PickScore Nodes
Image scoring nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) using [PickScore](https://github.com/yuvalkirstain/PickScore).<br>
Best used with a batch of images to check which ones fit a given prompt the best.

## Installation
Clone the repository to `custom_nodes` in your ComfyUI directory:
```
git clone https://github.com/Zuellni/ComfyUI-PickScore-Nodes
```

## Nodes
Name | Description
:--- | :---
Loader | Loads scoring models from Hugging Face or a given directory. The [default](https://huggingface.co/yuvalkirstain/PickScore_v1) one is around 4GB.
Processor | Takes images/text and converts them to embeddings.
Selector | Selects `count` best images/latents/masks. Interrupts generation if the `threshold` isn't met.