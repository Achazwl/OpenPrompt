import sys
sys.path.append(".")
sys.path.append("..")

from openprompt.utils.zh import num2zh

from openprompt.data_utils import InputExample
from openprompt.data_utils.ZH import E_reviews
# dataset tested on CPM2: #TODO
processor = E_reviews()
trainset = processor.get_train_examples("datasets/ZH/generation/E_reviews")
devset = processor.get_dev_examples("datasets/ZH/generation/E_reviews")

# from openprompt.data_utils.data_sampler import FewShotSampler
# sampler  = FewShotSampler(num_examples_per_label=64, num_examples_per_label_dev=64, also_sample_dev=True)
# trainset, devset = sampler(trainset, devset)

import bminf.torch as bt
"""
tutorial for using bminf
( using pip inside conda for enviroment management is preferable )
tested on: pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
1. git clone git@github.com:OpenBMB/BMInf.git
2. git checkout dev
3. pip install -r requirements.txt
4. python setup.py install
"""
use_cpm_version = 1
if use_cpm_version == 1:
    from openprompt.plms.lm import LMTokenizerWrapper
    plm = bt.models.CPM1()
    tokenizer = plm.tokenizer
    WrapperClass = LMTokenizerWrapper
elif use_cpm_version == 2:
    from openprompt.plms.seq2seq import CPM2TokenizerWrapper
    plm = bt.models.CPM2() # model size 11 G，downloading would take some time.
    # default cache path: ~/.cache/bigmodels/cpm2.1-new/
    tokenizer = plm.tokenizer
    WrapperClass = CPM2TokenizerWrapper
    
from openprompt.pipeline_base import PromptForGeneration as PromptForGeneration

from openprompt.prompts import SoftTemplate, MixedTemplate

mytemplate = SoftTemplate(
    model = plm,
    tokenizer = tokenizer,
    num_tokens = 100,
    text = '关键词:{"meta": "key_values", "shortenable": True} 目标：根据上述关键词信息，生成一段广告文案. 文案：{"mask"}',
)

wrapped_example = mytemplate.wrap_one_example(trainset[0]) 
print("Wrapped Example:", wrapped_example)

use_cuda = True
prompt_model = PromptForGeneration(plm=plm, template=mytemplate)
if use_cuda:
    prompt_model = prompt_model.cuda()

# ## below is standard training

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=trainset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256, 
    batch_size=8, shuffle=True, teacher_forcing=True, predict_eos_token=False,
    truncate_method="tail")
# next(iter(train_dataloader))

validation_dataloader = PromptDataLoader(dataset=devset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=256,
    batch_size=8, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

import torch
from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()

print("template name parameters: ", [n for n, p in prompt_model.template.named_parameters()])
optimizer_grouped_parameters = [
    {'params': [p for n,p in prompt_model.template.named_parameters()]}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-1/1024)

# flname = "loggg.txt"
# with open(flname, "w") as fl:
#     print(file=fl)

from openprompt.utils.metrics import generation_metric

for epoch in range(5):
    ## train
    prompt_model.train()

    tot_loss = 0 
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)*1024
        loss.backward()
        # print(prompt_model.template.soft_embeds.grad)
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {epoch} - step {step} / {len(train_dataloader)}: ", loss.item(), tot_loss/(step+1))
        break
    
    ## evaluate

    prompt_model = prompt_model.eval()

    generated_sentence = []
    groundtruth_sentence = []

    with torch.no_grad():
        for step, inputs in enumerate(validation_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            from IPython import embed; embed()
            output_sentence = prompt_model.generate(inputs)
            generated_sentence.extend(output_sentence)
            groundtruth_sentence.extend(inputs['tgt_text'])
        score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("val_score", score, flush=True)

    # with open(flname, "a") as fl:
    #     print("val accuracy:", acc, file=fl)