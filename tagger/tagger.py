import logging
import json
from typing import Generator
from pathlib import Path
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from tagger.interrogator import Interrogator
from tagger.interrogators import interrogators
from tagger.embedders import embedders

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class Tagger:
    DEFAULTS = {
        'model': 'wd14-convnextv2.v1',
        'embedder': 'sentence_transformer',
        'execution': 'cpu',
        'threshold': 0.35,
        'max_tags': 0,
        'rawtag': False,
        'compare_cpu': False,
        'tolerance': 1e-4
    }

    def __init__(self, *, model=None, embedder=None, execution=None, threshold=None, max_tags=None, rawtag=None, compare_cpu=None, tolerance=None):
        self.model = model or self.DEFAULTS['model']
        self.embedder_name = embedder or self.DEFAULTS['embedder']
        self.execution = execution or self.DEFAULTS['execution']
        self.threshold = threshold if threshold is not None else self.DEFAULTS['threshold']
        self.max_tags = max_tags if max_tags is not None else self.DEFAULTS['max_tags']
        self.rawtag = rawtag if rawtag is not None else self.DEFAULTS['rawtag']
        self.compare_cpu = compare_cpu if compare_cpu is not None else self.DEFAULTS['compare_cpu']
        self.tolerance = tolerance if tolerance is not None else self.DEFAULTS['tolerance']

        self.interrogator = interrogators[self.model]
        self.setup_execution_provider()
        self.setup_embedder()

    def setup_execution_provider(self):
        available_providers = ort.get_available_providers()
        provider_map = {
            'coreml': 'CoreMLExecutionProvider',
            'cuda': 'CUDAExecutionProvider',
            'tensorrt': 'TensorrtExecutionProvider',
            'cpu': 'CPUExecutionProvider'
        }
        requested_provider = provider_map[self.execution]
        
        if requested_provider in available_providers:
            self.interrogator.providers = [requested_provider]
            logging.info(f"Using {self.execution} for inference.")
        else:
            logging.warning(f"{self.execution} is not available. Falling back to CPU.")
            self.interrogator.use_cpu()

    def setup_embedder(self):
        if self.execution == 'cuda' and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        self.embedder = embedders[self.embedder_name](device=device)

    def run_inference(self, image_path, provider):
        original_provider = self.interrogator.providers[0]
        self.interrogator.providers = [provider]
        tags, ratings, embedding = self.image_interrogate(image_path)
        self.interrogator.providers = [original_provider]
        return tags, ratings, embedding

    def compare_results(self, result1, result2):
        tags1, ratings1, embedding1 = result1
        tags2, ratings2, embedding2 = result2
        
        tag_diff = set(tags1.keys()) ^ set(tags2.keys())
        if tag_diff:
            logging.warning(f"Tags differ: {tag_diff}")
        
        for tag in set(tags1.keys()) & set(tags2.keys()):
            if abs(tags1[tag] - tags2[tag]) > self.tolerance:
                logging.warning(f"Tag probability differs for {tag}: {tags1[tag]} vs {tags2[tag]}")
        
        for rating in ratings1.keys():
            if abs(ratings1[rating] - ratings2[rating]) > self.tolerance:
                logging.warning(f"Rating differs for {rating}: {ratings1[rating]} vs {ratings2[rating]}")
        
        embedding_diff = np.max(np.abs(embedding1 - embedding2))
        if embedding_diff > self.tolerance:
            logging.warning(f"Max embedding difference: {embedding_diff}")

    def image_interrogate(self, image_path: Path) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
        im = Image.open(image_path)
        result = self.interrogator.interrogate(im)

        tags = Interrogator.postprocess_tags(
            result[1],
            threshold=self.threshold,
            escape_tag=not self.rawtag,
            replace_underscore=not self.rawtag,
            exclude_tags=set())
        
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        top_tags = sorted_tags[:self.max_tags] if self.max_tags > 0 else sorted_tags
        tag_string = ", ".join([tag for tag, _ in top_tags])
        
        embedding = self.embedder.embed(tag_string)
        
        return tags, result[0], embedding

    def process_image(self, image: Image.Image) -> dict:
        """
        Process a PIL Image object and return the tags, ratings, and embedding.
        
        Args:
            image (PIL.Image.Image): The image to process.
        
        Returns:
            dict: A dictionary containing tags, ratings, and embedding.
        """
        result = self.interrogator.interrogate(image)

        tags = Interrogator.postprocess_tags(
            result[1],
            threshold=self.threshold,
            escape_tag=not self.rawtag,
            replace_underscore=not self.rawtag,
            exclude_tags=set())
        
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        top_tags = sorted_tags[:self.max_tags] if self.max_tags > 0 else sorted_tags
        tag_string = ", ".join([tag for tag, _ in top_tags])
        
        embedding = self.embedder.embed(tag_string)
        
        return {
            "tags": tags,
            "ratings": result[0],
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        }

    def process_file(self, file_path: str) -> str:
        """
        Process an image file and return the result as a JSON string.
        
        Args:
            file_path (str): Path to the image file.
        
        Returns:
            str: JSON string containing tags, ratings, and embedding.
        """
        image_path = Path(file_path)
        with Image.open(image_path) as img:
            result = self.process_image(img)
        
        if self.compare_cpu and self.interrogator.providers[0] != 'CPUExecutionProvider':
            with Image.open(image_path) as img:
                original_provider = self.interrogator.providers[0]
                self.interrogator.providers = ['CPUExecutionProvider']
                cpu_result = self.process_image(img)
                self.interrogator.providers = [original_provider]
            logging.info("Comparing results with CPU execution:")
            self.compare_results(result, cpu_result)
        
        logging.info("Processing complete. Output:")
        return json.dumps(result, indent=2, cls=NumpyEncoder)

    def process_directory(self, dir_path, ext, overwrite):
        root_path = Path(dir_path)
        for image_path in self.explore_image_files(root_path):
            caption_path = image_path.parent / f'{image_path.stem}{ext}'

            if caption_path.is_file() and not overwrite:
                logging.info(f'Skipping: {image_path}')
                continue

            logging.info(f'Processing: {image_path}')
            with Image.open(image_path) as img:
                result = self.process_image(img)

            tags_str = ', '.join(result['tags'].keys())

            with open(caption_path, 'w') as fp:
                fp.write(tags_str)

    @staticmethod
    def explore_image_files(folder_path: Path) -> Generator[Path, None, None]:
        for path in folder_path.iterdir():
            if path.is_file() and path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                yield path
            elif path.is_dir():
                yield from TaggerApp.explore_image_files(path)

# Export the defaults for use in Click
DEFAULTS = Tagger.DEFAULTS
