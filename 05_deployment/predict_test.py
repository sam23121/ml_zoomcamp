#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

loan_id = 'loan-123'
loan = {
    "job": "management", 
    "duration": 400, 
    "poutcome": "success"
}

loan_id_2 = 'loan-456'
client_2 = {"job": "student", "duration": 280, "poutcome": "failure"}


for loan_id, client in [(loan_id, loan), (loan_id_2, client_2)]:
    response = requests.post(url, json=client).json()
    print(response)

    if response['default']:
        print('This loan is likely to default: %s' % loan_id)
    else:
        print('This loan is not likely to default: %s' % loan_id)
