import sys
sys.path.append(".")
sys.path.append("..")

from openprompt.utils.zh import num2zh

from openprompt.data_utils import InputExample
from openprompt.data_utils.ZH import LCQMC
# dataset tested on CPM2: CMNLI, ChnSentiCorp, LCQMC
from openprompt.data_utils.data_sampler import FewShotSampler
processor = LCQMC()
trainset = processor.get_train_examples("datasets/ZH/paraphrase/LCQMC")
devset = processor.get_dev_examples("datasets/ZH/paraphrase/LCQMC")
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
use_cpm_version = 2
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

from openprompt.prompts import SoftTemplate, MixedTemplate

mytemplate = SoftTemplate(
    model = plm,
    tokenizer = tokenizer,
    num_tokens = 100,
    # text = '{"meta": "context", "shortenable": True} 上文中，{"meta": "entity"} 是一个{"mask"}。选项：{"meta": "options", "post_processing": lambda lis: ",".join([f"{i}:{choice}" for i, choice in enumerate(lis)])}',
    # text = '前提：{"meta": "premise", "shortenable": True} 假设: {"meta": "hypothesis", "shortenable": True} 问题：前提和假设是什么关系? 选项：{"meta": "options", "post_processing": lambda lis: ",".join([f"{i}:{choice}" for i, choice in enumerate(lis)])} 回答:{"mask"}',
    # text = '文本：{"meta": "context", "shortenable": True} 问题:上述文本所表达的情感是积极的还是消极的? 回答：{"mask"}',
    # text = '文章：{"meta": "text", "shortenable": True} 问题: {"meta": "question"} 选项：{"meta": "options", "post_processing": lambda lis: ";".join([f"{i}、{choice}" for i, choice in enumerate(lis)])} 回答:{"mask"}',
    # text = '释义：{"meta": "text"} 问题: 这句释义对应的古文是? 选项：{"meta": "options", "post_processing": lambda lis: ",".join([f"{i}:{choice}" for i, choice in enumerate(lis)])} 回答:{"mask"}',
    text = '句子一:{"placeholder": "text_a"} 句子二:{"placeholder": "text_b"} 问题：两个句子表达的意思相似吗？回答:{"mask"}'
)

wrapped_example = mytemplate.wrap_one_example(trainset[0]) 
print("Wrapped Example:", wrapped_example)

# ## Define the verbalizer
# In classification, you need to define your verbalizer, which is a mapping from logits on the vocabulary to the final label probability.

from openprompt.prompts import ManualVerbalizer
import torch

# for example the verbalizer contains multiple label words in each class
label_words = [str(l) for l in processor.get_labels()]
myverbalizer = ManualVerbalizer(tokenizer, num_classes=len(label_words), label_words=label_words, prefix = '')
print("Verbalizer token id:", myverbalizer.label_words_ids.data)

from openprompt import PromptForClassification

use_cuda = True
prompt_model = PromptForClassification(plm=plm, template=mytemplate, verbalizer=myverbalizer)
if use_cuda:
    prompt_model = prompt_model.cuda()

# ## below is standard training

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=trainset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=8, 
    batch_size=8, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
# next(iter(train_dataloader))

validation_dataloader = PromptDataLoader(dataset=devset, template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=256, decoder_max_length=8,
    batch_size=8, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

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

for epoch in range(5):
    # ## train
    prompt_model.train()

    tot_loss = 0 
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)*1024
        loss.backward()
        # print(prompt_model.template.soft_embeds.grad)
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        print(f"epoch {epoch} - step {step} / {len(train_dataloader)}: ", loss.item(), tot_loss/(step+1))
    
    # ## evaluate

    prompt_model = prompt_model.eval()

    allpreds = []
    alllabels = []
    with torch.no_grad():
        for step, inputs in enumerate(validation_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            logits = prompt_model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            print("val step :", step)

    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print("val accuracy:", acc)

    # with open(flname, "a") as fl:
    #     print("val accuracy:", acc, file=fl)