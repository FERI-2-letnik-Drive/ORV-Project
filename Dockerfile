FROM python:3.11

LABEL authors="Marcel"

WORKDIR /orv

COPY requirements-base.txt .
COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY . .
RUN chmod +x start.sh

EXPOSE 3002

CMD ["./start.sh"]