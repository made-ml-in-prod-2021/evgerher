import json
import pytest
from runner import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


@pytest.fixture
def correct_request():
  with open('resources/request.json', 'r') as fread:
    return json.load(fread)


# categorical_columns:
#   sex:
#     values: [0, 1]
#   cp:
#     values: [0, 1, 2, 3]
#   fbs:
#     values: [0, 1]
#   restecg:
#     values: [0, 1, 2]
#   exang:
#     values: [0, 1]
#   slope:
#     values: [0, 1, 2]
#   ca:
#     values: [0, 1, 2, 3, 4]
#   thal:
#     values: [0, 1, 2, 3]


@pytest.mark.parametrize(
    ['column', 'value'],
    [
      pytest.param('chol', 50, id='numeric [chol] underflow'),
      pytest.param('trestbps', 320.4, id='numeric [trestbps] overflow'),
      pytest.param('thal', 320.4, id='categorical [thal] invalid'),
      pytest.param('restecg', 3, id='categorical [restecg] invalid'),
    ]
)
def test_app_400_validation_error(column, value, correct_request, client):
  modified_request = dict(correct_request)
  idx = modified_request['columns'].index(column)
  modified_request['values'][idx] = value
  app_response = client.post('/predict', json=modified_request)
  assert app_response.status_code == 400, 'Expected column validation error'
  assert app_response.data.decode(app_response.charset).startswith(f'Invalid value for feature {column}')


def test_app_400_invalid_length(correct_request, client):
  modified_request = dict(correct_request)
  modified_request['columns'] = modified_request['columns'][:-1]
  modified_request['values'] = modified_request['values'][:-1]
  app_response = client.post('/predict', json=modified_request)
  assert app_response.status_code == 400, 'Expected validation error on number of columns'
  assert app_response.data.decode(app_response.charset).startswith('Invalid number of features: expected')


def test_app_400_unknown_column(correct_request, client):
  name = 'flammenwerfer'
  modified_request = dict(correct_request)
  modified_request['columns'][0] = name
  app_response = client.post('/predict', json=modified_request)
  assert app_response.status_code == 400, 'Expected error on unknown column'
  assert app_response.data.decode(app_response.charset) == f'Unknown feature {name}'


def test_app_200_predict(correct_request, client):
  app_response = client.post('/predict', json=correct_request)

  out = app_response.json
  assert 'label' in out and 'proba' in out, 'Expected two keys: `label` and `out`'
  assert out['label'] == (out['proba'] > 0.5), 'Threshold should be 0.5'
  assert app_response.status_code == 200, 'Expected success'

