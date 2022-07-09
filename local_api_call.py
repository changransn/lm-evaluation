from transformers import GPT2TokenizerFast, AutoModelForCausalLM
import torch

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",cache_dir="/import/ml-sc-nlpcheckpoints-scratch/changranh/cache/")
model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir="/import/ml-sc-nlpcheckpoints-scratch/changranh/cache/")

def local_api_call(text):
    # text_with_prefix = tokenizer.eos_token + text
    tokenized_text = tokenizer.tokenize(text)
    inputs = tokenizer(text,return_tensors="pt", return_offsets_mapping=True)
    input_ids = inputs.input_ids[0]
    input_ids = torch.cat((torch.tensor([tokenizer.eos_token_id]), input_ids))

    offset = [x[0] for x in inputs.offset_mapping.tolist()[0]]
    # model takes a [bs, ss] input
    logits = model(input_ids)[0]

    # input_ids eos tok1 tok2 tok3
    # logits lgt1 lgt2 lgt3 lgt4
    # And we want it to become
    # input_ids tok1 tok2 tok3
    # logits lgt1 lgt2 lgt3
    input_ids = input_ids[1:] # input_ids without the begining eos token
    logits = logits[:-1] # strip out the last logits


    logits_of_label = []
    for idx, input_id in enumerate(input_ids):
        logits_of_label.append(logits[idx][input_id].item())

    tokens = prepare_outputs(tokenized_text, offset, logits_of_label)
    return text, tokens

def prepare_outputs(tokenized_text, offset, logits_of_label):
    assert len(tokenized_text) == len(offset) == len(logits_of_label)
    d = {}
    for token, token_offset, logprob in zip(tokenized_text, offset, logits_of_label):
        d.append("token": token,
                "offset" token_offset:
                "logprob": logprob)
    return d


# text = 'How are you? I am fine'
# tokenized_text = ['How', 'Ġare', 'Ġyou', '?', 'ĠI', 'Ġam', 'Ġfine']
# logits_of_label [-42.27570343017578, -79.52528381347656, -66.69406127929688, -90.96470642089844, -97.17134094238281, -137.20448303222656, -109.66590881347656]
# offset [0, 4, 8, 11, 13, 15, 18]
# input_ids tensor([2437,  389,  345,   30,  314,  716, 3734])

