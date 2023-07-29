import json
from easydict import EasyDict as edict


def get_config_regression(
        model_name
):
    with open('config.json', 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams']
    dataset_args = config_all['datasetCommonParams']

    config = {}
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config = edict(config)

    return config
