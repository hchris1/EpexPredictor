FROM python:latest

WORKDIR /code/

COPY ./requirements.txt /code/requirements.txt
COPY ./predictor /code/predictor


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["fastapi", "run", "predictor/api/priceapi.py", "--port", "80"]
