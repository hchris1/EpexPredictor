FROM python:latest

WORKDIR /app/

COPY ./requirements.txt /app/
COPY ./predictor /app/predictor


RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip install --no-cache-dir --upgrade gunicorn


CMD ["gunicorn", "-b", "0.0.0.0:80", "-w", "1", "predictor.api.priceapi:app"]