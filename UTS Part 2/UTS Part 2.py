#Muhammad Yuda Pratama/21091397025

#inisialisasi numpy
#multiple neuron batch and multiple layer dengan layers 10

import numpy as np

#inisialisasi variabel dengan Input Layer 10 dan per Batchnya 6 Input
inputs = [[4.3, 7.6, 1.2, 8.6, 4.9, 2.0, 1.0, 0.9, 4.4, 1.4],
          [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.1],
          [0.1, 1.2, 1.3, 6.5, 7.2, 8.1, 5.0, 6.3, 5.2, 7.8],
          [0.4, 8.4, 5.9, 2.4, 5.6, 6.7, 2.3, 1.0, 8.4, 9.4],
          [0.5, 5.3, 6.4, 4.2, 1.9, 4.1, 4.3, 7.3, 1.0, 8.0],
          [0.6, 9.0, 4.6, 7.0, 3.0, 2.0, 2.3, 9.7, 7.2, 6.0]]

#panjang weights = panjang inputs(10) ; jumlah weights = jumlah neuron(5)
weights1 = [[0.35, -2.5, 6.3, 9.4, 9.3, 2.4, 5.0, 1.0, 8.8, -6.6],
           [3.4, 1.9, 4.5, 6.3, 7.3, -7.2, -8.6, 9.2, 0.9, 3.7],
           [0.5, -5.0, 5.5, 9.2, -5.1, 0.2, 0.7, -7.9, 5.0, -4.0],
           [8.4, 7.1, 1.5, 4.0, 7.0, 2.0, -2.8, 0.9, -4.1, 0.77],
           [-0.55, 4.0, 5.0, 4.5, -6.4, 3.0, -7.3, 4.6, -6.2, 0.45]]

#jumlah biases pada hidden layer1 = 5 neuron
biases1 = [8.0, 0.4, 3.0, 1.2, 9.3]

#panjang weights = neuron layer1(5) ; jumlah weights = jumlah neuron layer2(3)
weights2 = [[4.5, 2.1, 8.4, 5.6, 2.4],
            [6.5, 7.3, 2.1, 5.4, 9.0],
            [1.0, 3.8, 5.0, 6.8, 4.0]]

#jumlah biases pada hidden layer2 = 3 neuron
biases2 = [5, -3,-4.0]

#perintah untuk menghitung layer1 menggunakan inputs, weights1, dan biases1
layer1_outputs = np.dot(inputs, np.array(weights1).T) + biases1

#perintah untuk menghitung layer2 menggunakan hasil dari perhitungan pada layer1
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print output layer2
print(layer2_outputs)