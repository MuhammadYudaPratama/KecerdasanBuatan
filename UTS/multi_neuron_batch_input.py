# -*- coding: utf-8 -*-
"""Multi_Neuron_Batch_Input.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Bbs8WJBGDfXXR8isT_oxJlTeB4jLIgKl
"""

#Muhammad Yuda Pratama/21091397025

#inisialisasi numpy
#Multi Neuron Batch Input dengan Neuron 5
import numpy as np

#inisialisasi variabel dengan Input Layer 10 dan per Batchnya 6 Input
inputs = [[4.3, 7.6, 1.2, 8.6, 4.9, 2.0, 1.0, 0.9, 4.4, 1.4],
          [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.1],
          [0.1, 1.2, 1.3, 6.5, 7.2, 8.1, 5.0, 6.3, 5.2, 7.8],
          [0.4, 8.4, 5.9, 2.4, 5.6, 6.7, 2.3, 1.0, 8.4, 9.4],
          [0.5, 5.3, 6.4, 4.2, 1.9, 4.1, 4.3, 7.3, 1.0, 8.0],
          [0.6, 9.0, 4.6, 7.0, 3.0, 2.0, 2.3, 9.7, 7.2, 6.0]]

#panjang weights = panjang inputs(10) ; jumlah weights = jumlah neuron(5)
weights = [[0.35, -2.5, 6.3, 9.4, 9.3, 2.4, 5.0, 1.0, 8.8, -6.6],
           [3.4, 1.9, 4.5, 6.3, 7.3, -7.2, -8.6, 9.2, 0.9, 3.7],
           [0.5, -5.0, 5.5, 9.2, -5.1, 0.2, 0.7, -7.9, 5.0, -4.0],
           [8.4, 7.1, 1.5, 4.0, 7.0, 2.0, -2.8, 0.9, -4.1, 0.77],
           [-0.55, 4.0, 5.0, 4.5, -6.4, 3.0, -7.3, 4.6, -6.2, 0.45]]

#bias ada 5 angka karena Neuron 5 : Bias = Jumlah Neuron
biases = [8.0, 0.4, 3.0, 1.2, 9.3]

#ouputs
layer_outputs = np.dot(inputs, np.array(weights).T) + biases

#print ouputs
print(layer_outputs)