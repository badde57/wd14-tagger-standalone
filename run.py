import click
import logging
import sys
from tagger.tagger import Tagger, DEFAULTS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)

@click.command()
@click.option('--dir', help='Predictions for all images in the directory')
@click.option('--file', help='Predictions for one file')
@click.option('--threshold', type=float, default=DEFAULTS['threshold'], help='Prediction threshold (default is 0.35)')
@click.option('--ext', default='.txt', help='Extension to add to caption file in case of dir option (default is .txt)')
@click.option('--overwrite', is_flag=True, help='Overwrite caption file if it exists')
@click.option('--rawtag', is_flag=True, default=DEFAULTS['rawtag'], help='Use the raw output of the model')
@click.option('--model', default=DEFAULTS['model'], help='Model name to use for prediction')
@click.option('--embedder', default=DEFAULTS['embedder'], help='Embedder to use for generating embeddings from tags')
@click.option('--max-tags', type=int, default=DEFAULTS['max_tags'], help='Maximum number of tags to use for embedding (default is 0, meaning all tags)')
@click.option('--execution', type=click.Choice(['coreml', 'cuda', 'tensorrt', 'cpu']), 
              default=DEFAULTS['execution'], help='Execution provider to use for inference')
@click.option('--compare-cpu', is_flag=True, default=DEFAULTS['compare_cpu'], help='Compare results with CPU execution')
@click.option('--tolerance', type=float, default=DEFAULTS['tolerance'], help='Tolerance for result comparison')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']),
              default='INFO', help='Set the logging level')
def main(dir, file, threshold, ext, overwrite, rawtag, model, embedder, max_tags, execution, compare_cpu, tolerance, log_level):
    logging.getLogger().setLevel(log_level)

    if not dir and not file:
        raise click.UsageError("You must specify either --dir or --file.")

    app = Tagger(
        model=model,
        embedder=embedder,
        execution=execution,
        threshold=threshold,
        max_tags=max_tags,
        rawtag=rawtag,
        compare_cpu=compare_cpu,
        tolerance=tolerance
    )

    if file:
        result = app.process_file(file)
        print(result)

    if dir:
        app.process_directory(dir, ext, overwrite)

if __name__ == '__main__':
    main()
