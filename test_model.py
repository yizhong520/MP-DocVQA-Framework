import os, time, datetime
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.SP_DocVQA import singlepage_docvqa_collate_fn
from logger import Logger
from metrics import Evaluator
from utils import parse_args, time_stamp_to_hhmmss, load_config, save_json
from build_utils import build_model, build_dataset


def run_sample(data_loader, model):
    predictions = []
    for batch_idx, batch in enumerate(tqdm(data_loader)):
        bs = len(batch['question_id'])
        with torch.no_grad():
            outputs, pred_answers, pred_answer_page, answer_conf = model.forward(batch, return_pred_answer=True)

        print("REACHED HERE")

        predictions.append({
            "outputs": outputs,
            "pred_answers": pred_answers,
            "pred_answer_page": pred_answer_page,
            "answer_conf": answer_conf
        })

    return predictions


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    start_time = time.time()

    dataset = build_dataset(config, 'val')
    print("Dataset size:", len(dataset))
    
    val_data_loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=singlepage_docvqa_collate_fn)

    model = build_model(config)

    logger = Logger(config=config)
    logger.log_model_parameters(model)

    predictions = run_sample(val_data_loader, model)
    print(predictions)

