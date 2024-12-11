FROM agrigorev/model-2024-hairstyle:v3

COPY lambda_function.py .
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["lambda_function.lambda_handler"]