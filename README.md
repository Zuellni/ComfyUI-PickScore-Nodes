# ComfyUI PickScore Nodes
Image scoring nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) using [PickScore](https://github.com/yuvalkirstain/PickScore) to predict which images in a batch best fit a given prompt.
## Installation
Make sure your ComfyUI is up to date and clone the repository to `custom_nodes`:
```
git clone https://github.com/Zuellni/ComfyUI-PickScore-Nodes
```

## Nodes
Name | Description
:--- | :---
Loader | Loads scoring models from [Hugging Face](https://huggingface.co) or a given directory. The [default](https://huggingface.co/yuvalkirstain/PickScore_v1) one is around 4GB.
Processor | Takes images/text and converts them to inputs for the `Selector` node.
Selector | Selects up to `limit` best images/latents/masks. Interrupts processing if the `threshold` isn't reached.

## Workflow
The image below can be opened in ComfyUI.

![workflow](https://github.com/Zuellni/ComfyUI-PickScore-Nodes/assets/123005779/e8753778-7c54-418a-a6d8-a2a18fee0c2e)
