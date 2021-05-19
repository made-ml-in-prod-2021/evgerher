import requests
import json
import random


def swap(data, i, j):
  """
  Randomly swap two indexes
  """
  for field in ['columns', 'values']:
    col_tmp = data[field][i]
    data[field][i] = data[field][j]
    data[field][j] = col_tmp


def main():
  with open('resources/request.json', 'r') as fread:
    data = json.load(fread)

  N = len(data['columns'])

  # make a 100 requests, randomly swap order of columns
  # (proof it is invariant to order of columns)
  for idx in range(100):
    response = requests.post('http://localhost:5050/predict', json=data)
    # assert response.status_code == 200, 'Expected OK'


    print(f'[{idx + 1}]. {response.status_code}; {response.json()}')

    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    swap(data, i, j)

if __name__ == '__main__':
  main()
