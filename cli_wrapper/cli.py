import os
import json
import click

# TODO: support for results field


@click.command()
@click.argument('model-path')
@click.argument('input-path')
@click.argument('output-path')
@click.option('-rcf', '--round-conf', default=3, help='Number of decimal places to round confidence to')
@click.option('-rcd', '--round-coord', default=4, help='Number of decimal places to round bbox coordinates to')
@click.option('-rt', '--render-thresh', default=0.85, help='Minimum confidence value required to render a bbox')
@click.option('-ot', '--output-thresh', default=0.1, help='Minimum confidence value required to output a detection in the output file')
@click.option('--recursive/--not-recursive', default=False, help='Whether to search for images in folders within the base folder provided')
@click.option('-n', '--n-cores', default=0, help='Number of CPU cores to utilise. Will be ignored if a valid GPU is available')
@click.option('-cp', '--checkpoint-path', default=None, type=str, help='Path to JSON checkpoint file')
@click.option('-cf', '--checkpoint-frequency', default=-1, type=str, help='How often to write to checkpoint file, i.e.: every N images')
@click.option('--show/--no-show', default=False, help='Whether to output the results in the console')
@click.option('--bbox/--no-bbox', default=True, help='Whether save images with bounding boxes.')
@click.option('--verbose/--quiet', default=False, help='Whether to output or supress Tensorflow message')
def detect(model_path, input_path, output_path, round_conf, round_coord, render_thresh,
           output_thresh, recursive, n_cores, checkpoint_path,
           checkpoint_frequency, show, bbox, verbose):
    """Runs detection procedure on a set of images using a given
       MegaDetector model.


       MODEL_PATH: the path of the MegaDetector model file to use


       INPUT_PATH: the path of the image folder


       OUTPUT_PATH: path in which to save bbox images and JSON summary.
       """

    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    from tf_detector import TFDetector

    tf_detector = TFDetector(model_path=model_path,
                             output_path=output_path,
                             conf_digits=round_conf,
                             coord_digits=round_coord,
                             render_conf_threshold=render_thresh,
                             output_conf_threshold=output_thresh)

    results = tf_detector.run_detection(input_path=input_path,
                                        generate_bbox_images=bbox,
                                        recursive=recursive,
                                        n_cores=n_cores,
                                        results=None,
                                        checkpoint_path=checkpoint_path,
                                        checkpoint_frequency=checkpoint_frequency)

    if show:
        click.echo_via_pager(
            json.dumps(r, indent=4, default=str) for r in results)
