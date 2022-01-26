"""Test the webserver built with FastAPI"""

from fastapi.testclient import TestClient
from .. import main


client = TestClient(main.app)


def test_multivariate_lstm_train():
    """Tests for of explaining feature importance."""
    response = client.post(
        '/multivariate-lstm-train',
        json={
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
			"model_path": "../trained models/lstm/model",
			"scaler_path": "../trained models/lstm/scaler/scaler.gz"
		  },
		  "activation": "relu",
		  "optimizer": "adam",
		  "loss": "mae",
		  "nb_epochs": 10,
		  "batch_size": 64,
		  "validation_split": 0.15,
		  "initial_embeding_dim": 128,
		  "patience": 1
		}
    )

    assert response.status_code == 200



def test_aggregate_multivariate_lstm_score():
    """Tests for of explaining feature importance."""
    response = client.post(
        '/aggregate-multivariate-lstm-score',
        json= {
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
            "model_path": "../trained models/lstm/model",
            "scaler_path": "../trained models/lstm/scaler/scaler.gz"
          }
        }     
    )

    assert response.status_code == 200


# def test_aggregate_multivariate_lstm():
#     """Tests aggregattion of Multivariate data as reconstruction error of lstm."""
#     response = client.post(
#         '/aggregate-multivariate-lstm',
#         json={




#             'parameters': {
#                 'num_features': 32,
#                 'num_split': 3000
#             }
#         }
#     )
#     assert response.status_code == 200
#     assert response.json() == {
#         'univariate_time_series': {

#         }
#     }


# def test_aggregate_multivariate_var():
#     """Tests aggregattion of Multivariate data as reconstruction error of var."""
#     response = client.post(
#         '/aggregate-multivariate-var',
#         json={
#             'train_data': {

#             },
#             'score_data': {

#             },
#             'parameters': {
#                 'num_features': 32,
#                 'num_split': 3000
#             }
#         }
#     )
#     assert response.status_code == 200
#     assert response.json() == {
#         'univariate_time_series': {

#         }
#     }


# def test_aggregate_multivariate_pca():
#     """Tests aggregattion of Multivariate data as reconstruction error of pca."""
#     response = client.post(
#         '/aggregate-multivariate-pca',
#         json={
#             'train_data': {

#             },
#             'score_data': {

#             },
#             'parameters': {
#                 'num_features': 32,
#                 'num_split': 3000
#             }
#         }
#     )
#     assert response.status_code == 200
#     assert response.json() == {
#         'univariate_time_series': {

#         }
#     }
