import os
import sys
sys.path.append(".")

import argparse

from pytorch_lightning import Trainer as PLTrainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from openprompt.trainer import ClassificationRunner, GenerationRunner

from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt.utils.reproduciblity import set_seed
from openprompt import PromptDataLoader
from openprompt.prompts import load_template, load_verbalizer
from openprompt.data_utils import FewShotSampler
from openprompt.utils.logging import config_experiment_dir, init_logger, logger
from openprompt.config import get_yaml_config
from openprompt.plms import load_plm
from openprompt.data_utils import load_dataset
from openprompt.utils.utils import check_config_conflicts



def get_config():
    parser = argparse.ArgumentParser("classification config")
    parser.add_argument("--config_yaml", type=str, help='the configuration file for this experiment.')
    parser.add_argument("--resume", type=str, help='a specified checkpoint path to resume training, usually last.ckpt\
           It will fall back to run from initialization if no lastest checkpoint are found.')
    parser.add_argument("--test", type=str, help='a specified checkpoint path to test.')
    args = parser.parse_args()
    config = get_yaml_config(args.config_yaml)
    check_config_conflicts(config)
    logger.info("CONFIGS:\n{}\n{}\n".format(config, "="*40))
    return config, args


def build_dataloader(dataset, template, tokenizer, config, split):
    dataloader = PromptDataLoader(dataset=dataset, 
                                template=template, 
                                tokenizer=tokenizer, 
                                batch_size=config[split].batch_size,
                                shuffle=config[split].shuffle_data,
                                teacher_forcing=config[split].teacher_forcing \
                                    if hasattr(config[split],'teacher_forcing') else None,
                                predict_eos_token=True if config.task=="generation" else False,
                                **config.dataloader
                                )
    return dataloader

def save_config_to_yaml(config):
    from contextlib import redirect_stdout
    saved_yaml_path = os.path.join(config.logging.path, "config.yaml")
    with open(saved_yaml_path, 'w') as f:
        with redirect_stdout(f): print(config.dump())
    logger.info("Config saved as {}".format(saved_yaml_path))


def main():
    config, args = get_config()
    # init logger, create log dir and set log level, etc.
    if args.resume and args.test:
        raise ValueError("Cannot set --resume and --test arguments at the same time. ")
    elif args.resume or args.test:
        EXP_PATH = args.resume or args.test
        config.logging.path = None # TODO
    else:
        EXP_PATH = config_experiment_dir(config)
    
    init_logger(EXP_PATH+"/log.txt", config.logging.file_level, config.logging.console_level)
    # save config to the logger directory
    if not args.resume:
        save_config_to_yaml(config)
    # set seed
    set_seed(config)
    # load the pretrained models, its model, tokenizer, and config.
    plm_model, plm_tokenizer, plm_config = load_plm(config)
    # load dataset. The valid_dataset can be None
    train_dataset, valid_dataset, test_dataset, Processor = load_dataset(config) # TODO For effeciecy, load dataset needed only.
    
    if config.task == "classification":
        # define prompt
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        verbalizer = load_verbalizer(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config, classes=Processor.labels)
        # load prompt’s pipeline model
        prompt_model = PromptForClassification(plm_model, template, verbalizer)
    elif config.task == "generation":
        template = load_template(config=config, model=plm_model, tokenizer=plm_tokenizer, plm_config=plm_config)
        prompt_model = PromptForGeneration(plm_model, template, gen_config=config.generation)
    else:
        raise NotImplementedError(f"config.task {config.task} is not implemented yet. Only classification and generation are supported.")

    # process data and get data_loader
    if config.learning_setting == 'full':
        pass
    elif config.learning_setting == 'few_shot': # TODO make seed a list, run multiple times, get mean performance
        if config.few_shot.few_shot_sampling is not None:
            sampler = FewShotSampler(
                num_examples_per_label = config.sampling_from_train.num_examples_per_label,
                also_sample_dev = config.sampling_from_train.also_sample_dev,
                num_examples_per_label_dev = config.sampling_from_train.num_examples_per_label_dev
            )
            train_dataset, valid_dataset = sampler(
                train_dataset = train_dataset,
                valid_dataset = valid_dataset,
                seed = config.sampling_from_train.seed
            )
    elif config.learning_setting == 'zero_shot':
        pass # TODO without training
    
    train_dataloader = build_dataloader(train_dataset, template, plm_tokenizer, config, "train")
    valid_dataloader = build_dataloader(valid_dataset, template, plm_tokenizer, config, "dev")
    test_dataloader = build_dataloader(test_dataset, template, plm_tokenizer, config, "test")

    if config.task == "classification":
        model = ClassificationRunner(prompt_model, config)
    elif config.task == "generation":
        model = GenerationRunner(prompt_model, config)

    trainer = PLTrainer(
        gpus = config.environment.num_gpus,

        max_epochs = config.train.num_epochs,
        accumulate_grad_batches = config.train.gradient_accumulation_steps,

        gradient_clip_algorithm = "norm",
        gradient_clip_val = config.train.max_grad_norm,

        resume_from_checkpoint = args.resume, # TODO which path

        # enable_checkpointing = True, # TODO lightning version
        callbacks = [
            ModelCheckpoint(
                monitor = "val_metric", # TODO
                save_top_k = 1,
                save_last = True,
                mode = "max",

                dirpath = EXP_PATH + "/checkpoints",
                filename = "{epoch:02d}-{val_metric:.2f}", # TODO
            ),
        ],

        logger = TensorBoardLogger(
            save_dir = EXP_PATH + "/logger",
            name = "",
        ),
    )

    if args.test:
        trainer.test(test_dataloader = test_dataloader, ckpt_path = args.test) # TODO ckpt_path
    else:
        trainer.fit(model, train_dataloaders = train_dataloader, val_dataloaders = valid_dataloader)
        trainer.test(test_dataloaders = test_dataloader)

if __name__ == "__main__":
    main()