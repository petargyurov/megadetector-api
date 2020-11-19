import click
import json
from tf_detector import TFDetector

# TODO: support for results field
# TODO: verbose flag that controls logging output
# TODO: progress bar


@click.command()
@click.argument('model-path')
@click.argument('input-path')
@click.option('-rcf', '--round-conf', default=3, help='Number of decimal places to round confidence to')
@click.option('-rcd', '--round-coord', default=4, help='Number of decimal places to round bbox coordinates to')
@click.option('-rt', '--render-thresh', default=0.85, help='Minimum confidence value required to render a bbox')
@click.option('-ot', '--output-thresh', default=0.1, help='Minimum confidence value required to output a detection in the output file')
@click.option('-o', '--output-path', default=None, type=str, help='Path to JSON file to which to save the results to. If left blank results will not be saved')
@click.option('--recursive/--not-recursive', default=False, help='Whether to search for images in folders within the base folder provided')
@click.option('-n', '--n-cores', default=0, help='Number of CPU cores to utilise. Will be ignored if a valid GPU is available')
@click.option('-cp', '--checkpoint-path', default=None, type=str, help='Path to JSON checkpoint file')
@click.option('-cf', '--checkpoint-frequency', default=-1, type=str, help='How often to write to checkpoint file, i.e.: every N images')
@click.option('--show/--no-show', default=False, help='Whether to output the results in the console')
def detect(model_path, input_path, round_conf, round_coord, render_thresh,
		   output_thresh, output_path, recursive, n_cores, checkpoint_path,
		   checkpoint_frequency, show):

	"""Runs detection procedure on a set of images using a given
	   MegaDetector model.


	   MODEL_PATH: the path of the MegaDetector model file to use


	   INPUT_PATH: the path of the image folder
	   """

	tf_detector = TFDetector(model_path=model_path,
							 conf_digits=round_conf,
							 coord_digits=round_coord,
							 render_conf_threshold=render_thresh,
							 output_conf_threshold=output_thresh)

	results = tf_detector.run_detection(input_path=input_path,
										output_file=output_path,
										recursive=recursive,
										n_cores=n_cores,
										results=None,
										checkpoint_path=checkpoint_path,
										checkpoint_frequency=checkpoint_frequency)

	if show:
		click.echo_via_pager(json.dumps(r, indent=4, default=str) for r in results)
