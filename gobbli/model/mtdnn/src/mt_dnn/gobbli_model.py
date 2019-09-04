import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from .model import MTDNNModel


class GobbliMTDNNModel(MTDNNModel):
    def update(self, input_ids, token_type_ids, attention_mask, labels):
        self.network.train()
        if self.config['cuda']:
            labels = labels.cuda(async=True)

        y = Variable(labels, requires_grad=False)
        logits = self.mnetwork(input_ids, token_type_ids, attention_mask, task_id=0)
        loss = F.cross_entropy(logits, y)

        self.train_loss.update(loss.item(), logits.size(0))
        self.optimizer.zero_grad()

        loss.backward()
        if self.config['global_grad_clipping'] > 0:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                           self.config['global_grad_clipping'])

        self.optimizer.step()
        self.updates += 1
        self.update_ema()

    def predict(self, input_ids, token_type_ids, attention_mask):
        self.network.eval()
        score = self.mnetwork(input_ids, token_type_ids, attention_mask, task_id=0)
        score = F.softmax(score, dim=1).data.cpu()
        predict = np.argmax(score.numpy(), axis=1).tolist()
        return score, predict
