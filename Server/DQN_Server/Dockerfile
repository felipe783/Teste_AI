FROM python:3.12-slim
WORKDIR /Treinamento_DQN
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "DQN/train.py"]