from .nodes import ImageProcessor, Loader, Selector, TextProcessor

NODE_CLASS_MAPPINGS = {
    "ZuellniPickScoreLoader": Loader,
    "ZuellniPickScoreImageProcessor": ImageProcessor,
    "ZuellniPickScoreTextProcessor": TextProcessor,
    "ZuellniPickScoreSelector": Selector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZuellniPickScoreLoader": "PickScore Loader",
    "ZuellniPickScoreImageProcessor": "PickScore Image Processor",
    "ZuellniPickScoreTextProcessor": "PickScore Text Processor",
    "ZuellniPickScoreSelector": "PickScore Selector",
}
