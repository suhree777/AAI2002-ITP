import argparse
import time
import math
import os
import numpy as np
import torch
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM

parser = argparse.ArgumentParser(
    description='Evaluation script for Transformer Language Model')
parser.add_argument('--work_dir', default='LM-TFM', type=str,
                    help='experiment directory.')
parser.add_argument('--dataset', type=str, default='wt103',
                    choices=['wt103', 'lm1b', 'enwik8', 'text8', 'nesmdb'],
                    help='dataset name')
parser.add_argument('--tgt_len', type=int, default=70,
                    help='number of tokens to predict')
parser.add_argument('--eval_tgt_len', type=int, default=50,
                    help='number of tokens to predict for evaluation')
parser.add_argument('--batch_size', type=int, default=10,
                    help='batch size for evaluation')
args = parser.parse_args()

device = torch.device('cpu')

corpus = get_lm_corpus(args.work_dir, args.dataset)
ntokens = len(corpus.vocab)

model = MemTransformerLM(ntokens, 12, 10, 500, 50, 1000, 0.0, 0.0)
model_file = os.path.join(args.work_dir, 'model.pt')
model.load_state_dict(torch.load(model_file, map_location=device))
model = model.to(device)
model.eval()

eval_batch_size = args.batch_size
test_iter = corpus.get_iterator(
    'test', eval_batch_size, args.eval_tgt_len, device=device, ext_len=0)

total_loss = 0
total_len = 0
start_time = time.time()
with torch.no_grad():
    mems = tuple()
    for i, (data, target, seq_len) in enumerate(test_iter):
        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.float().mean().type_as(loss)
        total_loss += seq_len * loss.item()
        total_len += seq_len
    total_time = time.time() - start_time
    print('Test loss: {:.2f} | Test PPL: {:.2f} | Time: {:.2f}s'.format(
        total_loss / total_len, math.exp(total_loss / total_len), total_time))
