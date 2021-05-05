from argparse import ArgumentParser, Namespace
import logging

import pandas as pd

from ml_code.data import read_data
from ml_code.data.make_dataset import write_results
from ml_code.models import inference_model, train_model, SerializableArtifact, dump_artifact, load_artifact
from ml_code.utils import setup_logger, PipelineConfig

logger = logging.getLogger('runner')


def train_callback(args: Namespace):
  config: PipelineConfig = PipelineConfig.load_yaml(args.config_path)
  artifact: SerializableArtifact = train_model(config)
  dump_artifact(artifact, args.output_path)


def inference_callback(args: Namespace):
  artifact: SerializableArtifact = load_artifact(args.model_path)
  df: pd.DataFrame = read_data(args.data_path)
  results: pd.Series = inference_model(artifact, df, args.proba)
  write_results(results, args.output_path)


def parse_args():
  parser = ArgumentParser(
    prog='Classification pipeline runner',
    description='CLI tool to train the model and to predict results.'
  )

  subparsers = parser.add_subparsers(help='Select a command')
  train_parser = subparsers.add_parser('train', help='Train a model and save weights')
  train_parser.add_argument('--config',
                            required=True,
                            help='Path to the yaml config',
                            type=str,
                            metavar='/path/to/config.yaml',
                            dest='config_path')
  train_parser.add_argument('--output',
                            required=False,
                            help='Output path for the model weights (pickle)',
                            type=str,
                            metavar='/path/to/model.pkl',
                            default='model.pkl',
                            dest='output_path')
  train_parser.set_defaults(callback=train_callback)

  inference_parser = subparsers.add_parser('inference', help='Make an inference')
  inference_parser.add_argument('--model',
                                required=True,
                                help='Model weights path (pickle)',
                                type=str,
                                metavar='/path/to/model.pkl',
                                dest='model_path')
  inference_parser.add_argument('--data',
                                required=True,
                                help='Dataset to make an inference on (csv)',
                                metavar='/path/to/dataset.csv',
                                dest='data_path')
  inference_parser.add_argument('--output',
                                required=False,
                                default='output.csv',
                                metavar='/path/to/prediction.csv',
                                dest='output_path',
                                help='Provide path for inference results')
  inference_parser.add_argument('--proba',
                                required=False,
                                default=False,
                                action='store_true',
                                help='Whether to return probabilities or not')
  # todo: do I need an initial yaml file to handle an order of features when dealing with pandas?
  inference_parser.set_defaults(callback=inference_callback)
  return parser.parse_args()


def main():
  args = parse_args()
  args.callback(args)


if __name__ == '__main__':
  setup_logger()
  main()
