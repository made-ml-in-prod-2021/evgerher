# Model inference (web)

## Run locally 

`FLASK_APP=online_inference; flask run --host=0.0.0.0`

## Docker image

- Build `docker built -t evgerher/ml-in-prod-inference:0.0.1 .`  
- Run `docker run -it evgerher/ml-in-prod-inference`  

## API

Default port is 5050  
`curl -X POST -H "Content-Type: application/json" -d @resources/request.json localhost:5050`  



