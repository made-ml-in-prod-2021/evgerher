# Model inference (web)

## Run locally 

`python runner.py`

## Docker image

- Build `docker build -t evgerher/ml-in-prod-inference:0.0.1 .`  
  - The feature config is hardcoded `configs/health_features.yaml`  
  - The artifact path is hardcoded `artifact.pkl`  
- Pull image `docker pull evgerher/ml-in-prod-inference:0.0.1`
- Run `docker run -it -d -p 5050:5050 evgerher/ml-in-prod-inference:0.0.1`  

## API

Make an inference request via curl or python script (`requests` package is required)
Default port is 5050  

- `curl -X POST -H "Content-Type: application/json" -d @resources/request.json localhost:5050/predict` 
- `python knocker.py` 

## Tests


`python -m pytest`  

