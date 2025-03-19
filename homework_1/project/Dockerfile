FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV WANDB_API_KEY=your_wandb_api_key

CMD ["python", "train.py"]
