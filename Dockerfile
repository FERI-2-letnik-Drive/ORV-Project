FROM python:3.11

LABEL authors="Marcel"

WORKDIR /orv

COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x start.sh

EXPOSE 3002

CMD ["./start.sh"]