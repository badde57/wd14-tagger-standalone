from typing import Generator, Iterable
from tagger.interrogator import Interrogator
from tagger.embedders import embedders
from PIL import Image
from pathlib import Path
import click
import json
import numpy as np
import torch
import platform
import onnxruntime as ort
import logging
import sys

from tagger.interrogators import interrogators

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class MutuallyExclusiveOption(click.Option):
    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop('mutually_exclusive', []))
        help = kwargs.get('help', '')
        if self.mutually_exclusive:
            ex_str = ', '.join(self.mutually_exclusive)
            kwargs['help'] = help + (
                ' NOTE: This argument is mutually exclusive with '
                ' arguments: [' + ex_str + '].'
            )
        super(MutuallyExclusiveOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise click.UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(
                    self.name,
                    ', '.join(self.mutually_exclusive)
                )
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(
            ctx,
            opts,
            args
        )

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)

@click.command()
@click.option('--dir', cls=MutuallyExclusiveOption, mutually_exclusive=['file'],
              help='Predictions for all images in the directory')
@click.option('--file', cls=MutuallyExclusiveOption, mutually_exclusive=['dir'],
              help='Predictions for one file')
@click.option('--threshold', type=float, default=0.35, help='Prediction threshold (default is 0.35)')
@click.option('--ext', default='.txt', help='Extension to add to caption file in case of dir option (default is .txt)')
@click.option('--overwrite', is_flag=True, help='Overwrite caption file if it exists')
@click.option('--rawtag', is_flag=True, help='Use the raw output of the model')
@click.option('--recursive', is_flag=True, help='Enable recursive file search')
@click.option('--exclude-tag', multiple=True, help='Specify tags to exclude (can be used multiple times)')
@click.option('--model', default='wd14-convnextv2.v1', type=click.Choice(list(interrogators.keys())), help='Model name to use for prediction (default is wd14-convnextv2.v1)')
@click.option('--embedder', type=click.Choice(list(embedders.keys())), 
              default='gte', 
              help='Embedder to use for generating embeddings from tags')
@click.option('--max-tags', type=int, default=0, help='Maximum number of tags to use for embedding (default is 0, meaning all tags)')
@click.option('--execution', type=click.Choice(['coreml', 'cuda', 'tensorrt', 'cpu']), 
              default='cpu', help='Execution provider to use for inference')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
              default='WARNING', help='Set the logging level')
def main(dir, file, threshold, ext, overwrite, rawtag, recursive, exclude_tag, model, embedder, max_tags, execution, log_level):
    # Set the logging level based on the user's choice
    logging.getLogger().setLevel(log_level)

    if not dir and not file:
        raise click.UsageError("You must specify either --dir or --file.")
    
    # get interrogator configs
    interrogator = interrogators[model]

    # Set up the execution provider
    available_providers = ort.get_available_providers()
    provider_map = {
        'coreml': 'CoreMLExecutionProvider',
        'cuda': 'CUDAExecutionProvider',
        'tensorrt': 'TensorrtExecutionProvider',
        'cpu': 'CPUExecutionProvider'
    }

    requested_provider = provider_map[execution]
    
    if requested_provider in available_providers:
        interrogator.providers = [requested_provider]
        logging.info(f"Using {execution} for inference.")
    else:
        logging.warning(f"{execution} is not available. Falling back to CPU.")
        interrogator.use_cpu()

    # Initialize the embedder
    if execution == 'cuda' and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    embedder_model = embedders[embedder](device=device)

    def parse_exclude_tags() -> set[str]:
        if not exclude_tag:
            return set()

        tags = []
        for tag_str in exclude_tag:
            tags.extend([tag.strip() for tag in tag_str.split(',')])

        # reverse escape (nai tag to danbooru tag)
        reverse_escaped_tags = [tag.replace(' ', '_').replace('\(', '(').replace('\)', ')') for tag in tags]
        return set(tags + reverse_escaped_tags)  # reduce duplicates

    def image_interrogate(image_path: Path, tag_escape: bool, exclude_tags: Iterable[str]) -> tuple[dict[str, float], dict[str, float], np.ndarray]:
        """
        Predictions from an image path
        """
        im = Image.open(image_path)
        result = interrogator.interrogate(im)

        tags = Interrogator.postprocess_tags(
            result[1],
            threshold=threshold,
            escape_tag=tag_escape,
            replace_underscore=tag_escape,
            exclude_tags=exclude_tags)
        
        # Sort tags by probability
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        
        # If max_tags is 0 or greater than the number of tags, use all tags
        top_tags = sorted_tags[:max_tags] if max_tags > 0 else sorted_tags
        
        # Concatenate top tags into a single string
        tag_string = ", ".join([tag for tag, _ in top_tags])
        
        # Generate text embedding using the selected embedder
        embedding = embedder_model.embed(tag_string)
        
        return tags, result[0], embedding

    def explore_image_files(folder_path: Path) -> Generator[Path, None, None]:
        """
        Explore files by folder path
        """
        for path in folder_path.iterdir():
            if path.is_file() and path.suffix in ['.png', '.jpg', '.jpeg', '.webp']:
                yield path
            elif recursive and path.is_dir():
                yield from explore_image_files(path)

    if dir:
        root_path = Path(dir)
        for image_path in explore_image_files(root_path):
            caption_path = image_path.parent / f'{image_path.stem}{ext}'

            if caption_path.is_file() and not overwrite:
                # skip if caption exists
                logging.info(f'skip: {image_path}')
                continue

            logging.info(f'processing: {image_path}')
            tags = image_interrogate(image_path, not rawtag, parse_exclude_tags())

            tags_str = ', '.join(tags.keys())

            with open(caption_path, 'w') as fp:
                fp.write(tags_str)

    if file:
        tags, ratings, embedding = image_interrogate(Path(file), not rawtag, parse_exclude_tags())
        output = {
            "tags": tags,
            "ratings": ratings,
            "embedding": embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
        }
        print(json.dumps(output, indent=2, cls=NumpyEncoder))

if __name__ == '__main__':
    main()
