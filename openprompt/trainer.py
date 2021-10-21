import os
import sys
sys.path.append(".")

from torch.utils.data import dataloader

import pytorch_lightning as pl
from typing import Callable, Union
try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from tqdm import tqdm
import torch
from openprompt.prompts import *
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import  AdamW, get_linear_schedule_with_warmup
from openprompt.utils.calibrate import calibrate

class BasicRunner(pl.LightningModule):
    r"""The base class of All Runner
    common functions in Runners are defined in BasicRunner

    Args:
        prompt_model (:obj:`nn.Module`): the model to train
        config (:obj:`CfgNode`): A configuration object.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 config: CfgNode = None,
                ):
        super().__init__()
        self.model = model
        self.config = config
        self.automatic_optimization = False

    def on_save_checkpoint(self, checkpoint):
        print("save here")
        super().on_save_checkpoint(checkpoint)

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(
            limit_batches * batches)

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)
        effective_accum = num_devices * self.config.train.gradient_accumulation_steps
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        """num of training steps per epoch"""
        return self.num_training_steps // self.trainer.max_epochs
    
    def configure_optimizers(self):
        r"""config the optimizer and scheduler for
        
        1. model
        
        2. template
        
        3. verbalizer(optional)
        """
        
        optimizers = []

        if not self.config.plm.optimize.freeze_para:
            no_decay = self.config.plm.optimize.no_decay
            weight_decay = self.config.plm.optimize.weight_decay
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
                {'params': [p for n, p in self.model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]

            optimizer = {}
            optimizer["optimizer"] = AdamW(
                optimizer_grouped_parameters,
                lr = self.config.plm.optimize.lr,
                betas = self.config.plm.optimize.betas,
                eps = self.config.plm.optimize.eps
            )
            if self.config.plm.optimize.scheduler is not None:
                optimizer["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optimizer["optimizer"],
                        num_warmup_steps = self.config.plm.optimize.scheduler.num_warmup_steps, 
                        num_training_steps = self.num_training_steps
                    ),
                    "interval": "step",
                    "frequency": 1,
                }
            optimizers.append(optimizer)

        class Dummy:
            pass

        ## template_config 
        template_config = self.config[self.config.template]
        if hasattr(template_config, "optimize") and template_config.optimize is not None: # TODO should add optimize config in each yaml
            optimizer = {}
            if not hasattr(self.model.template, "optimize"):
                # using default gradient descent optimizer.
                optimizer["optimizer"] = AdamW(self.model.template.parameters(), lr = template_config.optimize.lr)
                if hasattr(template_config.optimize, "scheduler") and template_config.optimize.scheduler is not None:
                    optimizer["lr_scheduler"] = {
                        "scheduler": get_linear_schedule_with_warmup(
                            optimizer["optimizer"],
                            num_warmup_steps = template_config.optimize.scheduler.num_warmup_steps, 
                            num_training_steps = self.num_training_steps
                        ),
                        "interval": "step",
                        "frequency": 1,
                    }
            else:
                optimizer["optimizer"] = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(optimizer["optimizer"], "step", self.model.template.optimize)
                setattr(optimizer["optimizer"], "zero_grad", lambda:None)

                optimizer["optimizer"] = optimizer
            optimizers.append(optimizer)

        if hasattr(self.model, "verbalizer") and self.model.verbalizer:
            ## verbalizer_optimizer
            verbalizer_config = self.config[self.config.verbalizer]
            if hasattr(verbalizer_config, "optimize") and verbalizer_config.optimize is not None: # TODO should add verbalizer config in each yaml
                optimizer = {}
                if not hasattr(self.model.verbalizer, "optimize"):
                    # using default gradient descent optimizer.
                    optimizer["optimizer"] = AdamW(self.model.verbalizer.parameters(), lr = verbalizer_config.optimize.lr)
                    if hasattr(verbalizer_config.optimize, "scheduler") and verbalizer_config.optimize.scheduler is not None:
                        optimizer["lr_scheduler"] = {
                            "scheduler": get_linear_schedule_with_warmup(
                                optimizer["optimizer"],
                                num_warmup_steps = verbalizer_config.optimize.scheduler.num_warmup_steps, 
                                num_training_steps = self.num_training_steps
                            ),
                            "interval": "step",
                            "frequency": 1,
                        }
                else:
                    optimizer["optimizer"] = Dummy()
                    # resemble a pytorch optimizer for unified training.
                    setattr(optimizer["optimizer"], "step", self.model.verbalizer.optimize)
                    setattr(optimizer["optimizer"], "zero_grad", lambda:None)

                    optimizer["optimizer"] = optimizer
                optimizers.append(optimizer)

        return optimizers

    def accum_step(self, loss, batch_idx):
        loss = loss / self.config.train.gradient_accumulation_steps
        self.manual_backward(loss)
        if (batch_idx + 1) % self.config.train.gradient_accumulation_steps == 0:
            if self.config.train.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)
            for opt in self.optimizers():
                opt.step()
            for lr_scheduler in self.lr_schedulers():
                lr_scheduler.step()
            for opt in self.optimizers():
                opt.zero_grad()

    def training_epoch_end(self, outputs):
        for opt in self.optimizers():
            opt.zero_grad()

    def save_results(self, preds, tgts, split):
        ret_file_name = os.path.join(self.config.logging.path, f"{split}_preds.txt")
        with open(ret_file_name, 'w') as fout:
            for i in range(len(preds)):
                print(preds[i], file = fout)
        ret_file_name = os.path.join(self.config.logging.path, f"{split}_tgts.txt")
        with open(ret_file_name, 'w') as fout:
            for i in range(len(preds)):
                print(tgts[i], file = fout)

    def optimizers(self, use_pl_optimizer=False):
        opts = super().optimizers(use_pl_optimizer)
        return opts if isinstance(opts, list) else [opts]

    def lr_schedulers(self):
        schedulers = super().lr_schedulers()
        return schedulers if isinstance(schedulers, list) else [schedulers]


class ClassificationRunner(BasicRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer, 
    or self-training can use other runner class. 
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        prompt_model (:obj:`PromptForClassification`): One ``PromptModel`` object.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """
    def __init__(self, 
                 model: PromptForClassification,
                 config: CfgNode = None,
                 loss_function: Optional[Callable] = None,
                ):
        super().__init__(model, config)
        self.loss_function = loss_function if loss_function else self.config_loss_function()
    
    def config_loss_function(self, ):
        r"""config the loss function if it's not passed.
        """
        if self.config.classification.loss_function == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            return torch.nn.NLLLoss()
        else:
            raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        label = batch['label']
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()

    def validation_epoch_end(self, val_step_outputs):
        preds = []
        labels = []
        for pred, label in val_step_outputs:
            preds.extend(pred)
            labels.extend(label)

        self.save_results(preds, labels, split='val')
        
        scores = OrderedDict()
        for metric in self.config.classification.metric:
            score = classification_metrics(preds, labels, metric)
            scores[metric] = score
        self.log("val_metric", scores[self.config.classification.metric[0]]) # TODO metric (use which?)

    def test_step(self, batch, batch_idx):
        label = batch['label']
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()
    
    def test_epoch_end(self, test_step_outputs):
        preds = []
        labels = []
        for pred, label in test_step_outputs:
            preds.extend(pred)
            labels.extend(label)

        self.save_results(preds, labels, split='test')
        
        scores = OrderedDict()
        for metric in self.config.classification.metric:
            score = classification_metrics(preds, labels, metric)
            scores[metric] = score
        self.log("test_metric", scores)

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.loss_function(logits, batch['label'])
        self.log("trian_loss", loss, prog_bar = True, logger = True, on_step = True)
        self.accum_step(loss, batch_idx)

    def on_fit_start(self): # TODO how to run model that outside step
        if self.config.calibrate is not None:
            calibrate(self.model, self.config)

        verbalizer_config = self.config[self.config.verbalizer]
        template_config = self.config[self.config.template]
        if not hasattr(self.model.verbalizer, "optimize_to_initialize" ) and \
            not hasattr(self.model.template, "optimize_to_initialize" ):
            return None
        if hasattr(verbalizer_config, "init_using_split"):
            using_split = verbalizer_config.init_using_split
        elif hasattr(template_config, "init_using_split"):
            using_split = template_config.init_using_split
        else:
            using_split = "valid"

        if using_split == "train":
            dataloader = self.train_dataloader()
        elif using_split == "valid":
            dataloader = self.val_dataloader()
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
                batch = batch.to(self.device).to_dict()
                logits = self.model(batch)
            if hasattr(self.model.verbalizer, "optimize_to_initialize" ):
                self.model.verbalizer.optimize_to_initialize()
            if hasattr(self.model.template, "optimize_to_initialize" ):
                self.model.template.optimize_to_initialize()


class GenerationRunner(BasicRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer, 
    or self-training can use other runner class. 
    This class is specially implemented for generation.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptModel`` object.
        config (:obj:`CfgNode`): A configuration object.
    """
    def __init__(self, 
                 model: PromptForGeneration,
                 config: CfgNode = None,
                ):
        super().__init__(model, config)
    
    def validation_step(self, batch, batch_idx):
        target = batch['tgt_text']
        _, pred = self.model.generate(batch, **self.config.generation)
        return pred, target # these are already a cpu list

    def validation_epoch_end(self, val_step_outputs):
        preds = []
        targets = []
        for pred, target in val_step_outputs:
            preds.extend(pred)
            targets.extend(target)

        self.save_results(preds, targets, split='val')

        scores = OrderedDict()
        for metric in self.config.generation.metric:
            score = generation_metric(preds, targets, metric)
            scores[metric] = score
        self.log("val_metric", scores[self.config.generation.metric[0]]) # TODO metric which

    def test_step(self, batch, batch_idx):
        target = batch['tgt_text']
        _, pred = self.model.generate(batch, **self.config.generation)
        return pred, target # these are already a cpu list

    def test_epoch_end(self, val_step_outputs):
        preds = []
        targets = []
        for pred, target in val_step_outputs:
            preds.extend(pred)
            targets.extend(target)

        self.save_results(preds, targets, split='test')

        scores = OrderedDict()
        for metric in self.config.generation.metric:
            score = generation_metric(preds, targets, metric)
            scores[metric] = score
        self.log("test_metric", scores[self.config.generation.metric[0]]) # TODO metric which

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        self.log("training_loss", loss)
        return loss
