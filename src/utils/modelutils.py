import torch
import pdb
import torch
from transformers import BartTokenizer, BartForConditionalGeneration, utils

def remove_contraction_spaces(string):
    new_str = ''
    for i in range(len(string)):
        if string[i] == "'" and i+1 < len(string) and string[i+1].isalpha():
            new_str = new_str[:-1]+string[i]
        else:
            new_str += string[i] 
    return new_str


#takes in list of words and returns string
def join_words(words):
    string = ""
    for idx,word in enumerate(words):
        if word == "." or word == "," or word == "?" or word == "!" or word == "'s":
            string = string[:-1]+word+" "        
        elif word == "-":
            string = string[:-1]+word
        elif idx == len(words)-1:
            string += word            
        else:
            string += word+" "
    if string[-1] == " ":
        string = string[:-1]
    string = remove_contraction_spaces(string)
    return string




#takes in list of word labels, and returns indices correspondign to intrinsic errors
def get_intrinsic_indices(labels):
    return [ i for i in range(len(labels)) if labels[i] == 'intrinsic']

def find_consecutive_err_end_idx(start_idx,labels,error):
    end_idx = start_idx
    while end_idx < len(labels) and labels[end_idx] == error:
        end_idx += 1
    return end_idx


def space_before_needed(word_idx,words):
    if word_idx == 0 :
        return False
    prev_word = words[word_idx-1]
    word = words[word_idx]
    if prev_word == None or prev_word == "-" or word == "." or word == "," or word == "?" or word == "!" or word == "'s":
        return False
    return True

def join_word_range(words,start_idx,end_idx):
    if start_idx != end_idx:
        tok_str = join_words(words[start_idx:end_idx])
    else:
        tok_str = ''
    if space_before_needed(start_idx,words):
        tok_str = " "+tok_str
    return tok_str


#takes in a list of
def get_error_id_indices(labels,words,tokenizer,error):
    start_idx = 0
    end_idx = 0
    ids = []
    ids.append(torch.zeros(1,dtype=int))
    last_idx = 0
    intrinsic_id_indices = []
    while end_idx < len(words):
        if labels[end_idx] != error:
            end_idx += 1
        else:
            tok_str = join_word_range(words,start_idx,end_idx)
            if tok_str != '':
                str_ids = tokenizer(tok_str,return_tensors="pt",add_special_tokens=False).input_ids
                ids.append(str_ids[0])
                last_idx += len(str_ids[0])
            start_idx = end_idx
            end_idx = find_consecutive_err_end_idx(start_idx,labels,error)
            tok_str = join_word_range(words,start_idx,end_idx)
            str_ids = tokenizer(tok_str,return_tensors="pt",add_special_tokens=False).input_ids
            ids.append(str_ids[0])
            intrinsic_id_indices.extend(list(range(last_idx+1,last_idx+1+len(str_ids[0]))))
            start_idx = end_idx
    if start_idx < end_idx:
        tok_str = join_word_range(words,start_idx,end_idx)
        str_ids = tokenizer(tok_str,return_tensors="pt",add_special_tokens=False).input_ids
        ids.append(str_ids[0])        
    ids.append(torch.tensor([2],dtype=int))
    ids = torch.cat(ids)
    return intrinsic_id_indices




#takes in two strings, one for the encoder and decoder
def get_input_ids(words,tokenizer):

    return tokenizer(words, return_tensors="pt",max_length=1024, add_special_tokens=True,truncation=True).input_ids



#takes in list of list of input_ids and extracts out list of input_ids
def extract_1d_id_list(input_id_list):
    if input_id_list.dim() > 1:
        return input_id_list[0]
    else:
        return input_id_list

#takes in encoder input ids and decoder input ids as produced by tokenizer (that is a list of list) and returns
#list of words corresponding to input ids
def get_id_text(ids,tokenizer):
    ids = extract_1d_id_list(ids)
    text = tokenizer.convert_ids_to_tokens(ids)
    text = [sub.replace('Ġ','␣') for sub in text] 
    return text 

def generate_ref_sequences(input_ids,embed,tokenizer):
    ref_input_ids = torch.zeros_like(input_ids)
    ref_input_ids[0,0] = tokenizer.bos_token_id
    ref_input_ids[0,1:-1] = tokenizer.pad_token_id
    ref_input_ids[0,-1] = tokenizer.eos_token_id
    ref_input_embeds = embed(ref_input_ids)
    return ref_input_embeds

def replace_special_bart_tokens(text_list):
    text_list = [sub.replace('<s>',"begin_sequence") for sub in text_list]    
    text_list = [sub.replace('</s>',"end_sequence") for sub in text_list]
    return text_list

def get_tokens_from_text(text,tokenizer):
    token_ids = get_input_ids(text,tokenizer)
    token_text = get_tokens_from_ids(token_ids,tokenizer)    
    return token_text

def get_tokens_from_ids(token_ids,tokenizer):
    token_text = replace_special_bart_tokens(get_id_text(token_ids,tokenizer))    
    return token_text   

def bart_forward_func(attr_embeds,other_input_embeds,bart,pred_idx,index,attr_mode="enc"):
    if attr_mode=="enc":
        outputs = bart(inputs_embeds=attr_embeds,decoder_inputs_embeds=other_input_embeds)
    elif attr_mode=="dec":
        outputs = bart(inputs_embeds=other_input_embeds,decoder_inputs_embeds=attr_embeds)
    pred = outputs.logits[:,index-1,pred_idx]
    return pred





def init_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name, output_attentions=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model,tokenizer,device