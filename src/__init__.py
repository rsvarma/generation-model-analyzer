import os
from flask import Flask, render_template, redirect, request, session
from flask_session import Session
from .utils.modelutils import *
import torch
import pdb
from captum.attr import IntegratedGradients

def create_app(test_config=None):
    #create and configure the app
    app= Flask(__name__)
    app.secret_key='dev'
    app.config["SESSION_PERMANENT"]=False
    app.config["SESSION_TYPE"] = "filesystem"    
    Session(app)
    app.jinja_env.globals.update(zip=zip)  

    with app.app_context():

        from .home import home
        from .analyze_token import analyze_token

        app.register_blueprint(home.home_bp)
        app.register_blueprint(analyze_token.analyze_bp)
        
    return app
