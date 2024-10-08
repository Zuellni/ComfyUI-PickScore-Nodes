# ComfyUI PickScore Nodes
Image scoring nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) using [PickScore](https://github.com/yuvalkirstain/PickScore) to predict which images in a batch best match a given prompt.

## Installation
Clone the repository to `custom_nodes`:
```
git clone https://github.com/Zuellni/ComfyUI-PickScore-Nodes custom_nodes/ComfyUI-PickScore-Nodes
```

## Nodes
Name | Description
:--- | :---
Loader | Loads scoring models from [Hugging Face](https://huggingface.co) or a given directory. Uses [yuvalkirstain/PickScore_v1](https://huggingface.co/yuvalkirstain/PickScore_v1) by default.
Processor | Takes images/text and converts them to inputs for the `Selector` node.
Selector | Selects up to `limit` images and passes them to other nodes along with latents/masks of the same shape. Interrupts processing if the `threshold` isn't reached.

## Workflow
An example workflow is embedded in the image below and can be opened in ComfyUI.

![Workflow](https://github.com/Zuellni/ComfyUI-PickScore-Nodes/assets/123005779/9f439d31-c3cc-4e06-b650-eb2e102344e6)
