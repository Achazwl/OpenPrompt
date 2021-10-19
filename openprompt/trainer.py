import os
import sys
sys.path.append(".")

from torch.utils.data import dataloader

import pytorch_lightning as pl
from typing import Callable, OrderedDict, Union
from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from tqdm import tqdm
import torch
from openprompt.prompts import *
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import  AdamW, get_linear_schedule_with_warmup
from openprompt.utils.calibrate import calibrate

class ClassificationRunner(pl.LightningModule):
    def __init__(self, 
                 model: PromptForClassification,
                 config: CfgNode = None,
                 loss_function: Optional[Callable] = None,
                ):
        super().__init__()

        self.model = model
        self.config = config
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

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    @property
    def steps_per_epoch(self) -> int:
        return self.num_training_steps // self.trainer.max_epochs
    
    def configure_optimizers(self):
        r"""config the optimizer and scheduler for 1. model 2. template 3. verbalizer
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
            optimizers.append({})
            optimizers[-1]["optimizer"] = AdamW(
                optimizer_grouped_parameters,
                lr = self.config.plm.optimize.lr,
                betas = self.config.plm.optimize.betas,
                eps = self.config.plm.optimize.eps
            )
            if self.config.plm.optimize.scheduler is not None:
                optimizers[-1]["lr_scheduler"] = get_linear_schedule_with_warmup(
                    optimizers[-1]["optimizer"],
                    num_warmup_steps = self.config.plm.optimize.scheduler.num_warmup_steps, 
                    num_training_steps = self.num_training_steps
                )

        class Dummy:
            pass

        ## template_config 
        template_config = self.config[self.config.template]
        if hasattr(template_config, "optimize") and template_config.optimize is not None:
            optimizers.append({})
            if not hasattr(self.model.template, "optimize"):
                # using default gradient descent optimizer.
                optimizers[-1]["optimizer"] = AdamW(self.model.template.parameters(), lr = template_config.optimize.lr)
                if hasattr(template_config.optimize, "scheduler") and template_config.optimize.scheduler is not None:
                    optimizers[-1]["lr_scheduler"] = get_linear_schedule_with_warmup(
                        optimizers[-1]["optimizer"], 
                        num_warmup_steps = template_config.optimize.scheduler.num_warmup_steps, 
                        num_training_steps = self.num_training_steps
                    )
            else:
                optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(self.template_optimizer, "step", self.model.template.optimize)
                setattr(self.template_optimizer, "zero_grad", lambda:None)

                optimizers[-1]["optimizer"] = optimizer

        if self.model.verbalizer:
            ## verbalizer_optimizer
            verbalizer_config = self.config[self.config.verbalizer]
            if hasattr(verbalizer_config, "optimize") and verbalizer_config.optimize is not None:
                optimizers.append({})
                if not hasattr(self.model.verbalizer, "optimize"):
                    # using default gradient descent optimizer.
                    optimizers[-1]["optimizer"] = AdamW(self.model.verbalizer.parameters(), lr = verbalizer_config.optimize.lr)
                    if hasattr(verbalizer_config.optimize, "scheduler") and verbalizer_config.optimize.scheduler is not None:
                        optimizers[-1]["lr_scheduler"] = get_linear_schedule_with_warmup(
                            optimizers[-1]["optimizer"], 
                            num_warmup_steps = verbalizer_config.optimize.scheduler.num_warmup_steps, 
                            num_training_steps = self.num_training_steps
                        )
                else:
                    optimizer = Dummy()
                    # resemble a pytorch optimizer for unified training.
                    setattr(self.verbalizer_optimizer, "step", self.model.verbalizer.optimize)
                    setattr(self.verbalizer_optimizer, "zero_grad", lambda:None)

                    optimizers[-1]["optimizer"] = optimizer

        return optimizers
    
    def validation_step(self, batch, batch_idx):
        label = batch['label']
        logits = self.model(batch)
        pred = torch.argmax(logits, dim=-1)
        return pred.cpu().tolist(), label.cpu().tolist()

    def validation_epoch_end(self, val_step_outputs, log_name="val_metric"):
        preds = []
        labels = []
        for pred, label in  val_step_outputs:
            preds.extend(pred)
            labels.extend(label)
        
        scores = OrderedDict()
        for metric in self.config.classification.metric:
            score = classification_metrics(preds, labels, metric)
            scores[metric] = score
        self.log(log_name, scores["micro-f1"]) # TODO use which metric (use shell to eval instead?)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, test_step_outputs):
        self.validation_epoch_end(test_step_outputs, log_name="test_metric")

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        loss = self.loss_function(logits, batch['label'])
        return loss


    def on_fit_start(self): # TODO how to run model that outside step
        pass
    #     if self.config.calibrate is not None:
    #         calibrate(self.model, self.config)

    #     verbalizer_config = self.config[self.config.verbalizer]
    #     template_config = self.config[self.config.template]
    #     if not hasattr(self.model.verbalizer, "optimize_to_initialize" ) and \
    #         not hasattr(self.model.template, "optimize_to_initialize" ):
    #         return None
    #     if hasattr(verbalizer_config, "init_using_split"):
    #         using_split = verbalizer_config.init_using_split
    #     elif hasattr(template_config, "init_using_split"):
    #         using_split = template_config.init_using_split
    #     else:
    #         using_split = "valid"

    #     if using_split == "train":
    #         dataloader = self.train_dataloader
    #     elif using_split == "valid":
    #         dataloader = self.valid_dataloader
    #     else:
    #         raise NotImplementedError

    #     with torch.no_grad():
    #         for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
    #             batch = batch.to(self.device).to_dict()
    #             logits = self.model(batch)
    #         if hasattr(self.model.verbalizer, "optimize_to_initialize" ):
    #             self.model.verbalizer.optimize_to_initialize()
    #         if hasattr(self.model.template, "optimize_to_initialize" ):
    #             self.model.template.optimize_to_initialize()


class GenerationRunner(ClassificationRunner): # TODO
    pass
#     r"""A runner for simple training without training tricks.
#     Applying training tricks such as ensemble of template or verbalizer, 
#     or self-training can use other runner class. 
#     This class is specially implemented for generation.

#     Args:
#         prompt_model (:obj:`Union[DataParallel, PromptForClassification]`): One ``PromptModel`` object.
#         train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
#         valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
#         test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
#         config (:obj:`CfgNode`): A configuration object.
#     """
#     def __init__(self, 
#                  prompt_model: Union[DataParallel, PromptForGeneration],
#                  train_dataloader: Optional[PromptDataLoader] = None,
#                  valid_dataloader: Optional[PromptDataLoader] = None,
#                  test_dataloader: Optional[PromptDataLoader] = None,
#                  config: CfgNode = None,
#                  ):
#         super().__init__(prompt_model=prompt_model,
#                          train_dataloader=train_dataloader,
#                          valid_dataloader=valid_dataloader,
#                          test_dataloader=test_dataloader,
#                          config=config)
    
#     def config_loss_function(self,):
#         r""" No need to config loss_function in generation.
#         """
#         pass
    
#     def config_optimize(self,):
#         r"""config the optimizer and scheduler for 1. model 2. template 3. verbalizer
        
#         """
        
#         self.train_steps_per_epoch = len(self.train_dataloader) // self.config.train.gradient_accumulation_steps
#         num_training_steps = self.train_steps_per_epoch * self.config.train.num_epochs

#         if not self.config.plm.optimize.freeze_para:
#             no_decay = self.config.plm.optimize.no_decay
#             weight_decay = self.config.plm.optimize.weight_decay
#             optimizer_grouped_parameters = [
#                 {'params': [p for n, p in self.inner_model.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': weight_decay},
#                 {'params': [p for n, p in self.inner_model.model.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
#             ]

#             self.model_optimizer = AdamW(optimizer_grouped_parameters, lr = self.config.plm.optimize.lr)
#             if self.config.plm.optimize.scheduler is not None:
#                 self.model_scheduler = get_linear_schedule_with_warmup(
#                     self.model_optimizer, 
#                     num_warmup_steps = self.config.plm.optimize.scheduler.num_warmup_steps, 
#                     num_training_steps = num_training_steps
#                 )
#             else:
#                 self.model_scheduler = None
#         else:
#             self.model_optimizer = None
#             self.model_scheduler = None


#         class Dummy:
#             pass

#         ## template_config 
#         template_config = self.config[self.config.template]
#         if template_config.optimize is not None:
#             if not hasattr(self.inner_model.template, "optimize"):
#                 # using default gradient descent optimizer.
#                 no_decay = template_config.optimize.no_decay
#                 weight_decay = template_config.optimize.weight_decay
#                 optimizer_grouped_parameters = [
#                     {'params': [p for n, p in self.inner_model.template.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],'weight_decay': weight_decay},
#                     {'params': [p for n, p in self.inner_model.template.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],'weight_decay': 0.0}
#                 ]

#                 self.template_optimizer = AdamW(self.inner_model.template.parameters(), 
#                                                 lr = template_config.optimize.lr,
#                                                 betas = template_config.optimize.betas,
#                                                 eps = template_config.optimize.eps)
#                 if hasattr(template_config.optimize, "scheduler") and template_config.optimize.scheduler is not None:
#                     self.template_scheduler = get_linear_schedule_with_warmup(
#                         self.template_optimizer, 
#                         num_warmup_steps = template_config.optimize.scheduler.num_warmup_steps, 
#                         num_training_steps = num_training_steps
#                     )
#                 else:
#                     self.template_scheduler = None
#             else:
#                 self.template_optimizer = Dummy()
#                 # resemble a pytorch optimizer for unified training.
#                 setattr(self.template_optimizer, "step", self.inner_model.template.optimize)
#                 setattr(self.template_optimizer, "zero_grad", lambda:None)
#                 self.verbalizer_scheduler = None
#         else:
#             self.template_optimizer = None
#             self.template_scheduler = None
#         self.optimizers = [self.model_optimizer, self.template_optimizer]
#         self.schedulers = [self.model_scheduler, self.template_scheduler]

#     def evaluate(self, dataloader, split, post_evaluate_hook=None):
#         ret_file_name= os.path.join(self.config.logging.path,"{}_generated_text.txt".format(split))
        
#         tgt_texts = []
#         generated_sentences_all = []
#         for batch in tqdm(dataloader, desc=split):
#             batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
#             output_sequences, generated_sentences = self.inner_model.generate(batch, **self.config.generation)
#             tgt_texts.extend(batch['tgt_text'])
#             generated_sentences_all.extend(generated_sentences)
            
#         fout = open(ret_file_name,'w')
#         for i in range(len(generated_sentences_all)):
#             fout.write(generated_sentences_all[i]+"\n")
#         fout.close()

#         scores = OrderedDict()
#         scores_str = ""
#         for metric in self.config.generation.metric:
#             score = generation_metric(generated_sentences_all, tgt_texts, metric)
#             scores[metric] = score
#             scores_str += "{}: {}\n".format(metric, score)
#         logger.info("{} Performance: {}".format(split, scores_str.strip()))
#         return scores

#     def train_epoch(self, epoch):
#         self.prompt_model.train()
#         self.prompt_model.zero_grad()
#         total_loss = 0.0
#         sum_loss = 0.0
#         pbar = tqdm(self.train_dataloader, desc="Train epoch {}".format(epoch))
#         for step, batch in enumerate(pbar):
#             batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
#             loss = self.prompt_model(batch).mean()  #TODO：unbanlanced batch chunks
#             if self.config.train.gradient_accumulation_steps > 1:
#                 loss = loss / self.config.train.gradient_accumulation_steps
#             sum_loss += loss.item()
#             loss.backward()

#             if (step+1) % self.config.train.gradient_accumulation_steps == 0:
#                 pbar.set_postfix({ 'loss': sum_loss })
#                 if self.config.train.max_grad_norm > 0:
#                     torch.nn.utils.clip_grad_norm_(self.prompt_model.parameters(), self.config.train.max_grad_norm)
#                 for optimizer in self.optimizers:
#                     if optimizer is not None:
#                         optimizer.step()

#                 for scheduler in self.schedulers:
#                     if scheduler is not None:
#                         scheduler.step()

#                 for optimizer in self.optimizers:
#                     if optimizer is not None:
#                         optimizer.zero_grad()
#                 total_loss += sum_loss
#                 sum_loss = 0.
#         logger.info("Epoch {}, avg_loss: {:.4f}, total_loss: {:.4f}".format(epoch, total_loss / self.train_steps_per_epoch, total_loss))
#         return total_loss
