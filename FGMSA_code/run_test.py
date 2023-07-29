import gc
import os
import re
import time
import torch
import argparse
import pandas as pd

from config import get_config_regression
from data_loader import MMDataLoader
from models import AMIO
from trains import ATIO
from utils import assign_gpu

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

target_map = {'acc2': 'Acc-2', 'acc3': 'Acc-3', 'acc5': 'Acc-5', 'f1': 'F1', 'mae': 'MAE'}


def run(
        model_path: str,
):
    model_name = ''
    if 'lf-dnn' in model_path:
        model_name = 'lf_dnn'
    elif 'tfn' in model_path:
        model_name = 'tfn'
    elif 'lmf' in model_path:
        model_name = 'lmf'
    num_workers = 4

    target = re.search(r'-([^-]+)\.pth', model_path).group(1)
    args = get_config_regression(target)
    args['model_name'] = model_name
    args['dataset_name'] = 'fmsa'
    args['device'] = assign_gpu([0])
    args['train_mode'] = 'regression'
    torch.cuda.set_device(args['device'])

    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args)
    model.load_state_dict(torch.load(model_path))
    model.to(args['device'])

    trainer = ATIO().getTrain(args)
    results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    df = pd.DataFrame(columns=list(['Acc-2', 'Acc-3', 'Acc-5', 'F1', 'MAE']))
    df.loc[0] = list(map(lambda x: str(round(x * 100, 2)), results.values()))[:-2]
    df.loc[0, target_map[target]] = df.loc[0, target_map[target]] + '(state-of-the-art)'
    print(df.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model target.')
    parser.add_argument('model_path',
                        choices=['./models_trained/fg-mlf-dnn-acc2.pth', './models_trained/fg-mlf-dnn-f1.pth',
                                 './models_trained/fg-mlf-dnn-mae.pth', './models_trained/fg-mtfn-acc3.pth',
                                 './models_trained/fg-mtfn-acc5.pth'],
                        help='Trained models, always in "models_trained" folder.')
    args = parser.parse_args()

    run(args.model_path)
