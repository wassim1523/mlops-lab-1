FROM python:3.14-slim
RUN pip install --no-cache-dir mlflow==3.16.1
EXPOSE 5000
CMD ["mlflow","server","--host","0.0.0.0","--port","5000","--allowed-hosts","mlflow:5000,host.docker.internal:5000,localhost:5000,127.0.0.1:5000","--backend-store-uri","sqlite:////mlflow-data/mlflow.db","--artifacts-destination","/mlflow-data/artifacts"]
