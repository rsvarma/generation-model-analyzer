import os
from flask import Blueprint,Flask, render_template, redirect, request, session
from flask_session import Session
from ..utils.modelutils import *
import torch
import pdb
from captum.attr import IntegratedGradients

home_bp = Blueprint(
    'home_bp',__name__,
    template_folder='templates',
    static_folder='static'
)

@home_bp.route('/prepend',methods=['POST'])
def prepend_tokens():
    tokens = request.form['text'].split(',')
    tokens = [int(token) for token in tokens]
    dec_ids = session['dec_ids'][0].tolist()
    start_idx = session['prepend_token_len']
    dec_ids = tokens+dec_ids[start_idx:]
    dec_ids = torch.tensor(dec_ids).unsqueeze(0)
    dec_text = replace_special_bart_tokens(get_id_text(dec_ids,session['tokenizer']))
    session['dec_ids'] = dec_ids
    session['prepend_token_len'] = len(tokens)
    session['dec_token_text'] = dec_text
    return render_home() 

@home_bp.route('/generate',methods=['POST'])
def generate_summary():
    enc_ids = session['enc_ids']

    dec_ids = session.get('dec_ids')
    encoder_text = session['enc_token_text']
    model = session['model']
    tokenizer= session['tokenizer']
    device = session['device'] 
    if dec_ids is not None:
        #get all decoder ids but dont include end sequence
        dec_ids = dec_ids[:,:-1]
        session['dec_ids'] = model.generate(input_ids=enc_ids.to(device),decoder_input_ids=dec_ids.to(device)).cpu()
        session['dec_text'] = tokenizer.batch_decode(session['dec_ids'],skip_special_tokens=True)[0]
        session['dec_token_text'] = get_tokens_from_ids(session['dec_ids'],tokenizer)  
    else:
        session['dec_ids'] = model.generate(enc_ids.to(device)).cpu()
        session['dec_text'] = tokenizer.batch_decode(session['dec_ids'],skip_special_tokens=True)[0]
        session['dec_token_text'] = get_tokens_from_ids(session['dec_ids'],tokenizer)
    return render_home() 


@home_bp.route('/',methods=['GET'])
def index_get():
    if session.get('model') == None or session.get('tokenizer') == None or session.get('device') == None:
        print("running init_model")
        #for refactoring to all models will have to take in code for init_model
        session['model'],session['tokenizer'],session['device'] = init_model() 
    return render_home()

@home_bp.route('/',methods=['POST'])
def index_post():
    text = request.form['text']
    if text == "":
        text = None 
        token_text = None 
        token_ids = None 
    else:
        token_text = get_tokens_from_text(text,session['tokenizer'])
        token_ids = get_input_ids(text,session['tokenizer'])
    if 'enc_submit' in request.form:
        session['enc_text'] = text
        session['enc_token_text'] = token_text
        session['enc_ids'] = token_ids
    elif 'dec_submit' in request.form:
        session['dec_text'] = text
        session['dec_token_text'] = token_text 
        session['dec_ids'] = token_ids
        session['prepend_token_len'] = 1
    return render_home()



def render_home():
    enc_text = session['enc_text'] if session.get('enc_text') is not None else None 
    dec_text = session['dec_text'] if session.get('dec_text') is not None else None
    enc_token_text = session['enc_token_text'] if session.get('enc_token_text') is not None else None
    dec_token_text = session['dec_token_text'] if session.get('dec_token_text') is not None else None  
    return render_template('home.jinja2',
            enc_text = enc_text,
            dec_text = dec_text,
            enc_token_text=enc_token_text,
            dec_token_text=dec_token_text,
            prepend_button = True)