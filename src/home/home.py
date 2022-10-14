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
    enc_token_text = session['enc_token_text'] if session.get('enc_token_text') is not None else None
    dec_token_text = session['dec_token_text'] if session.get('dec_token_text') is not None else None       
    return render_template('home.jinja2',
            enc_token_text=enc_token_text,
            dec_token_text=dec_token_text)    



@home_bp.route('/',methods=['GET'])
def index_get():
    if session.get('model') == None or session.get('tokenizer') == None or session.get('device') == None:
        print("running init_model")
        #for refactoring to all models will have to take in code for init_model
        session['model'],session['tokenizer'],session['device'] = init_model() 
    enc_token_text = session['enc_token_text'] if session.get('enc_token_text') is not None else None
    dec_token_text = session['dec_token_text'] if session.get('dec_token_text') is not None else None       
    return render_template('home.jinja2',
            enc_token_text=enc_token_text,
            dec_token_text=dec_token_text)

@home_bp.route('/',methods=['POST'])
def index_post():
    text = request.form['text']
    token_text = get_tokens_from_text(text,session['tokenizer'])

    token_ids = get_input_ids(text,session['tokenizer'])
    if 'enc_submit' in request.form:
        session['enc_token_text'] = token_text
        session['enc_ids'] = token_ids
    elif 'dec_submit' in request.form:
        session['dec_token_text'] = token_text 
        session['dec_ids'] = token_ids
        session['prepend_token_len'] = 1
    enc_token_text = session['enc_token_text'] if session.get('enc_token_text') is not None else None
    dec_token_text = session['dec_token_text'] if session.get('dec_token_text') is not None else None  
    return render_template('home.jinja2',
            enc_token_text=enc_token_text,
            dec_token_text=dec_token_text)

        
