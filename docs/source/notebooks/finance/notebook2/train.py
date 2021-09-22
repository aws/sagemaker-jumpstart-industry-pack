from __future__ import print_function

import ast
import argparse
import warnings
import subprocess
import logging
import os
import pandas as pd
import numpy as np
import json
import time
from timeit import default_timer as timer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    from autogluon.tabular import TabularDataset, TabularPredictor

# ------------------------------------------------------------ #
# Training                                                     #
# ------------------------------------------------------------ #


def _load_input_data(path: str) -> TabularDataset:
    
    input_data_files = os.listdir(path)

    try:
        input_dfs = [pd.read_csv(f'{path}/{data_file}') for data_file in input_data_files]

        return TabularDataset(data=pd.concat(input_dfs))
    except:
        logger.info("No csv data in %s", path)
        return None


def train(args):
    
    # Load training and validation data
    logger.info("Train files: %s", os.listdir(args.train))
    train_data = _load_input_data(args.train)
    test_data = _load_input_data(args.test)
    
    
    # Train models
    args.init_args['path'] = args.model_dir
    
    if args.fit_args:
        predictor = TabularPredictor(
            **args.init_args
            ).fit(train_data, **args.fit_args)
    else:
        predictor = TabularPredictor(
            **args.init_args
            ).fit(train_data)
    
    logger.info("Best model: %s", predictor.get_model_best())
    
    # Leaderboard
    lb = predictor.leaderboard()
    lb.to_csv(f'{args.output_data_dir}/leaderboard.csv', index=False)
    logger.info("Saved leaderboard to output.")
    
    # Feature importance
    feature_importance = predictor.feature_importance(test_data)
    feature_importance.to_csv(f'{args.output_data_dir}/feature_importance.csv')
    logger.info("Saved feature importance to output.")
    
    # Evaluation
    evaluation = predictor.evaluate(test_data)
    with open(f'{args.output_data_dir}/evaluation.json', 'w') as f:
        json.dump(evaluation, f)
    logger.info("Saved evaluation to output.")
    
    predictor.save_space()
    
# ------------------------------------------------------------ #
# Inference                                                    #
# ------------------------------------------------------------ #
    test_data_nolabel = test_data.drop(labels=args.init_args['label'], axis=1)
    y_pred = predictor.predict(test_data_nolabel)
    y_pred.to_csv(f'{args.output_data_dir}/predictions.csv', index=False)

    
    return    


# ------------------------------------------------------------ #
# Training execution                                           #
# ------------------------------------------------------------ #

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variable
    # https://github.com/aws/sagemaker-training-toolkit/blob/master/ENVIRONMENT_VARIABLES.md#
    parser.add_argument("--output-data-dir", type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TESTING'])
    # Arguments to be passed to TabularPredictor()
    parser.add_argument('--init_args', type=lambda s: ast.literal_eval(s),
                        default="{'label': 'Rating'}",
                        help='https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor')
    # Arguments to be passed to TabularPredictor fit() method
    parser.add_argument('--fit_args', type=lambda s: ast.literal_eval(s),
                        help='https://auto.gluon.ai/stable/_modules/autogluon/tabular/predictor/predictor.html#TabularPredictor')

    return parser.parse_args()



if __name__ == '__main__':
    
    start = timer()
    args = parse_args()

    # Verify label is included
    if 'label' not in args.init_args:
        raise ValueError('"label" is a required parameter of "init_args"!')   
        
    # Convert optional fit call hyperparameters from strings
    if args.fit_args:
        if 'hyperparameters' in args.fit_args:
            for model_type, options in args.fit_args['hyperparameters'].items():
                assert isinstance(options, dict)
                for k,v in options.items():
                    args.fit_args['hyperparameters'][model_type][k] = eval(v) 
                
    # Print SageMaker args
    if args.fit_args:
        logger.info("fit_args:")
        for k, v in args.fit_args.items():
            logger.info("%s, type: %s, value: %s", k, type(v), v)
        
    train(args)

    elapsed_time = round(timer()-start, 3)
    logger.info("Elapsed time: %d seconds. Training Completed!", elapsed_time)

