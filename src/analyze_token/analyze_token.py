import os
from flask import Blueprint,Flask, render_template, redirect, request, session
from flask_session import Session
from ..utils.modelutils import *
import torch
import pdb
from captum.attr import IntegratedGradients

analyze_bp = Blueprint(
    'analyze_bp',__name__,
    url_prefix='/analyze',
    template_folder='templates',
    static_folder='static'
)

@analyze_bp.route('/<int:dec_idx>', methods=('GET', 'POST'))
def analyze(dec_idx):
    enc_ids = session['enc_ids']
    dec_ids = session['dec_ids'][:,:dec_idx]
    encoder_text = session['enc_token_text']
    decoder_text = session['dec_token_text']
    model = session['model']
    tokenizer= session['tokenizer']
    device = session['device']      



    outputs = model(input_ids=enc_ids.to(device), decoder_input_ids=dec_ids.to(device))

    #get probability values        
    logits = outputs.logits.detach().cpu()
    softmax = torch.nn.Softmax(dim=0)
    prob_scores = softmax(logits[0,-1])
    pred_vals,pred_inds = torch.sort(prob_scores,descending=True)
    pred_text = get_id_text(pred_inds,tokenizer)
    pred_info = zip(pred_vals[:1000].tolist(),pred_text[:1000])

    #get attention values
    cross_attentions_list = [x.detach().cpu() for x in outputs.cross_attentions]  
    decoder_attentions_list = [x.detach().cpu() for x in outputs.decoder_attentions]
    encoder_att_vals = []
    encoder_ind_vals = []
    encoder_text_vals = []
    encoder_attr_titles = []                         
    for layer_id,cross_attentions in enumerate(cross_attentions_list):
        layer_score = torch.mean(cross_attentions[:,:,dec_idx-1,:].squeeze(0),dim=0)
        top_info = torch.sort(layer_score,descending=True)
        encoder_att_vals.append(top_info.values.tolist())
        encoder_ind_vals.append(top_info.indices.tolist())
        encoder_text_vals.append([encoder_text[idx] for idx in top_info.indices.tolist()])
        encoder_attr_titles.append(f"Layer {layer_id+1}")
    encoder_att_info = zip(encoder_att_vals,encoder_ind_vals,encoder_text_vals,encoder_attr_titles)    

    decoder_att_vals = []
    decoder_ind_vals = []
    decoder_text_vals = []     
    decoder_attr_titles = []     
    for layer_id,decoder_attentions in enumerate(decoder_attentions_list):
        layer_score = torch.mean(decoder_attentions[:,:,dec_idx-1,:].squeeze(0),dim=0)
        top_info = torch.sort(layer_score,descending=True)
        decoder_att_vals.append(top_info.values.tolist())
        decoder_ind_vals.append(top_info.indices.tolist())
        decoder_text_vals.append([decoder_text[idx] for idx in top_info.indices.tolist()])      
        decoder_attr_titles.append(f"Layer {layer_id+1}")  
    decoder_att_info = zip(decoder_att_vals,decoder_ind_vals,decoder_text_vals,decoder_attr_titles)



    #get integrated gradient values    
    embed = torch.nn.Embedding(model.model.shared.num_embeddings,model.model.shared.embedding_dim,model.model.shared.padding_idx)
    embed.weight.data = model.model.shared.weight.data

    ref_encoder_embeds = generate_ref_sequences(enc_ids,embed.cpu(),tokenizer)
    ref_decoder_embeds = generate_ref_sequences(dec_ids,embed.cpu(),tokenizer)

    encoder_embeds = embed(enc_ids)
    decoder_embeds = embed(dec_ids)




    pred_idx = session['dec_ids'][0,dec_idx]


    ig = IntegratedGradients(bart_forward_func)

    ig_attributions = []
    ig_inds = []
    ig_tokens = []
    ig_titles = []



    encoder_input_attributions = ig.attribute(inputs=encoder_embeds.to(device),baselines = ref_encoder_embeds.to(device),additional_forward_args= (decoder_embeds.to(device),model,pred_idx,dec_idx),n_steps=300,internal_batch_size=32).detach().cpu()
    encoder_input_attributions = encoder_input_attributions.sum(dim=-1)
    encoder_input_attributions = encoder_input_attributions/torch.norm(encoder_input_attributions) 
    encoder_input_attributions,encoder_attr_inds = torch.sort(encoder_input_attributions[0],descending=True) 
    encoder_text_attr = [encoder_text[idx] for idx in encoder_attr_inds]
    ig_attributions.append(encoder_input_attributions.tolist())
    ig_inds.append(encoder_attr_inds.tolist())
    ig_tokens.append(encoder_text_attr)
    ig_titles.append("Encoder Input Attributions")
    torch.cuda.empty_cache()         

    


    decoder_attributions = ig.attribute(inputs=decoder_embeds.to(device),baselines=ref_decoder_embeds.to(device),additional_forward_args=(encoder_embeds.to(device),model,pred_idx,dec_idx,"dec"),n_steps=300,internal_batch_size=32).detach().cpu()
    decoder_attributions = decoder_attributions.sum(dim=-1)
    decoder_attributions = decoder_attributions/torch.norm(decoder_attributions)
    decoder_attributions,decoder_attr_inds = torch.sort(decoder_attributions[0],descending=True)
    decoder_text_attr = [decoder_text[idx] for idx in decoder_attr_inds]
    ig_attributions.append(decoder_attributions.tolist())
    ig_inds.append(decoder_attr_inds.tolist())
    ig_tokens.append(decoder_text_attr)        
    ig_titles.append("Decoder Attributions")
    torch.cuda.empty_cache()      

    ig_info = zip(ig_attributions,ig_inds,ig_tokens,ig_titles)  

    return render_template('analyze.jinja2',sel_token=decoder_text[dec_idx],
                            pred_info=pred_info,
                            enc_info=encoder_att_info,
                            dec_info=decoder_att_info,
                            ig_info=ig_info)
