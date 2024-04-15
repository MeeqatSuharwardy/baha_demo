"""
Main module of flask API.
"""
# Third party modules
import os
import glob
import io
import base64
from typing import Any
from functools import wraps
import pandas as pd
from flask import (
    Flask, request,
    json, make_response, Response
)
# from flask_cors import CORS, cross_origin
from generator import generate
from asgiref.wsgi import WsgiToAsgi
# Module
app = Flask(__name__)
# cors = CORS(app)
asgi_app = WsgiToAsgi(app)
api_cors = {
  "origins": ["*"],
  "methods": ["OPTIONS", "GET", "POST"],
  "allow_headers": ["Content-Type"]
}
#app.config['PROPAGATE_EXCEPTIONS'] = True
# upload_folder = 'uploads'
# os.makedirs(upload_folder, exist_ok = True)


def validate_request(request_api: Any)-> bool:
    """
    method will take a json request and perform all validations if the any error
    found then return error response with status code if data is correct then
    return data in a list.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    bool
        return True or False.

    """

    if "data" in request_api.files:
        return True
    if "data" not in request_api.files:
        return False
def validate_text_request(request_api: Any)-> bool:
    """
    method will take a json request and perform all validations if the any error
    found then return error response with status code if data is correct then
    return data in a list.

    Parameters
    ----------
    request_api: Request
        contain the request data in file format.

    Return
    ------
    bool
        return True or False.

    """
    data = request_api.get_json()
    if "data" in data:
        if data["data"] == '':
            return False
        return True
    if "data" not in data:
        return False
def get_textdata(data: json)-> str:
    """
    method will take a json data and return text_data.

    Parameters:
    ----------
    data: json
        json data send in request.

    Return:
    ------
    text_data: str
        return text_data as string.

    """
    text_data = data["data"]
    return text_data

def get_data(request_api: Any)-> str:
    """
    method will take request and get data from request then return thhe data.

    Parameters:
    ----------
    request_api: Request
        contain the request data in file format.

    Return:
    ------
    image_file: str
        return the data file as string.

    """
    data = request_api.files["data"]
    return data
def make_bad_params_value_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'data key error',
        'category' : 'Bad Params',}),
        400)
    return result
def make_file_save_error_response()-> Response:
    """
    method will make a error response a return it back.

    Parameters:
    ----------
    None

    Return:
    ------
    Response
        return a response message.

    """
    result = make_response(json.dumps(
        {'message'  : 'File not save sucesfully',
        'category' : 'Bad Params error',}),
        400)
    return result

@app.route('/floor_plan_pretarin', methods = ['POST'])
# @cross_origin(**api_cors)
def generate_floor_plan():
    """
    method will take the text prompt as input and return the generated reponse.

    Parameters:
    ----------
    None

    Return:
    ------
    str
        return the reponse.

    """
    try:
        if validate_text_request(request):
            query = request.get_json()
            text_data = get_textdata(query)
            text_data = "flat top-down occupancy-grid style view of an architectural floorplan of "+ text_data +". Two-dimensional view, black and white"
            img = generate(
                prompt = text_data,
                negative_prompt = "Aspect Ratio 1:1, complex, 3D, 2D",
                seed = 785245150,
                width = 1024,
                height = 1024,
                guidance_scale_base = 11.7,
                guidance_scale_refiner = 5.0,
                num_inference_steps_base = 25,
                num_inference_steps_refiner = 20,
                apply_refiner = False
                )
            img_byte_array = io.BytesIO()
            img.save(img_byte_array, format=img.format)
            img_byte_array = img_byte_array.getvalue()
            # Convert bytes to base64-encoded string
            img_base64_string = base64.b64encode(img_byte_array).decode('utf-8')
            output = {
                "prompt": text_data,
                "image" : img_base64_string
            }
            return Response(
                json.dumps(output),
                mimetype = 'application/json'
                )
        return make_bad_params_value_response()
    except Exception as exception:
        result = make_response(json.dumps(
                    {'message'  : str(exception),
                    'category' : 'Internal server error',}),
                    500)
        return result
if __name__=='__main__':
    app.run(debug = True, host = "0.0.0.0")
