# Multivariate Aggregator Module

## Build
```sh
$ docker build . -t multivariate-aggregator
```

## Run
```sh
$ docker run -p 8080:8080 -v mydata:/app multivariate-aggregator
```

## Documentation
* Swagger: http://localhost:8080/documentation
* ReDoc: http://localhost:8080/redoc

## Use


### 1. multivariate-lstm-train

#### Curl
```js
curl -X 'POST' \
  'http://localhost:8080/multivariate-lstm-train' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "train_data": {
    "data": {
		"A1": [601.929 , 587.4339, 590.4059, 596.6575, 582.4339, 585.8266, 597.0676, 584.5786, 583.95  , 595.0085],
		"A2": [650.8372, 650.8372, 650.8372, 650.309 , 650.309 , 649.7221, 649.1147, 649.6167, 649.6167, 649.6167],
		"A3": [636.3697, 636.3697, 636.3697, 636.3697, 636.3697, 636.3697, 636.3697, 636.3697, 636.3697, 636.3697],
		"A4": [ 71.1788,  71.1788,  71.4192,  70.8146,  71.2311,  70.9744, 70.9744,  71.1484,  71.9672,  71.509 ],
		"A5": [ 36.9295,  36.9295,  37.1119,  36.722 ,  36.97  ,  36.8511, 36.8511,  36.9359,  37.4204,  37.1334]
		}
  },
  "paths": {
    "model_path": "src/trained models/lstm/model",
    "scaler_path": "src/trained models/lstm/scaler/scaler.gz"
  },
  "activation": "relu",
  "optimizer": "adam",
  "loss": "mae",
  "nb_epochs": 10,
  "batch_size": 64,
  "validation_split": 0.15,
  "initial_embeding_dim": 128,
  "patience": 1
}'
```

#### Response Body
```js
"model is saved successfully"
```


### 2. aggregate-multivariate-lstm-score

#### Curl
```js
curl -X 'POST' \
  'http://localhost:8080/aggregate-multivariate-lstm-score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "test_data": {
    "data": {
            "A1": [580.7722, 592.1779, 587.5173, 583.7109, 594.7249, 604.5849, 611.5132, 616.7466, 608.9669, 597.9345],
            "A2": [649.4124, 649.4124, 649.4124, 650.1096, 651.0769, 651.632 , 652.3653, 652.3653, 652.3653, 652.7337],
            "A3": [636.3428, 636.3428, 636.3428, 635.6159, 635.6159, 635.9999, 635.9999, 635.9999, 636.6101, 636.6101],
            "A4": [ 71.9601,  71.9601,  72.342 ,  73.5115,  74.2349,  73.8276, 73.5101,  73.2902,  72.4169,  72.7627],
            "A5": [ 37.4148,  37.4148,  37.5577,  38.3091,  38.7071,  38.4878, 38.3124,  38.1843,  37.69  ,  37.8794]
		}
  },
  "paths": {
    "model_path": "src/trained models/lstm/model",
    "scaler_path": "src/trained models/lstm/scaler/scaler.gz"
  }
}'
```
#### Response body
```js
{
  "score": [
    0.4473551023156773,
    0.5246782198868474,
    0.5840548551021769,
    1.1894471926386436,
    1.6542528848245126,
    1.6095836563318535,
    1.6604827630411783,
    1.6393310930729599,
    1.2404790148631542,
    1.2843145189487302
  ]
}
```

### 2. VAR

### 3. PCA


## New Release
1. Update `__version__` in `src/main.py` with a new commit.
2. Tag this commit.
