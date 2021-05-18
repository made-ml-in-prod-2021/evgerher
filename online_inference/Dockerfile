FROM python:3.8-slim

WORKDIR /app
ADD . /app
RUN pip install -r requirements.txt
#ENV PORT 5050
EXPOSE 5050/tcp
CMD ["ls", "-alh"]
CMD ["python", "runner.py"]
