import os

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# exec(open("/home/pi/micmon/dataset/Dataset.py").read())
# exec(open("/home/pi/micmon/model/Model.py").read())
from micmon.dataset.Dataset import Dataset
from micmon.model.Model import Model

# This is a directory that contains the saved .npz dataset files
datasets_dir = os.path.expanduser('datasets/sound-detect/data')

# This is the output directory where the model will be saved
model_dir = os.path.expanduser('models/sound-detect')

# This is the number of training epochs for each dataset sample
epochs = 5

# Load the datasets from the compressed files.
# 65% of the data points will be included in the training set,
# 35% of the data points will be included in the evaluation set
# and used to evaluate the performance of the model.
datasets = Dataset.scan(datasets_dir, validation_split=0.35)
labels = ['Faulty', 'Not faulty']
freq_bins = len(datasets[0].samples[0])

# Create a network with 4 layers (one input layer, two intermediate layers and one output layer).
# The first intermediate layer in this example will have twice the number of units as the number
# of input units, while the second intermediate layer will have 75% of the number of
# input units. We also specify the names for the labels and the low and high frequency range
# used when sampling.
model = Model(
    [
        layers.Input(shape=(freq_bins,)),
        layers.Dense(int(2 * freq_bins), activation='relu'),
        layers.Dense(int(0.75 * freq_bins), activation='relu'),
        layers.Dense(2, activation='softmax'),
    ],
    labels=labels,
    low_freq=datasets[0].low_freq,
    high_freq=datasets[0].high_freq
)

# Train the model
for epoch in range(epochs):
    loss1 = []
    Accuracy1 = []
    for i, dataset in enumerate(datasets):
        print(f'[epoch {epoch + 1}/{epochs}] [audio sample {i + 1}/{len(datasets)}]')
        model.fit(dataset)
        evaluation = model.evaluate(dataset)
        loss1.append(evaluation[0])
        Accuracy1.append(evaluation[1])
        print(f'Validation set loss and accuracy: {evaluation}')

# Save the model
LossData = np.array(loss1)
AccuracyData = np.array(Accuracy1)
fig, plot1 = plt.subplots(1, 2, figsize=(10, 10))
#plot1[0].title('Training Loss')
#plot1[1].title('Training Accuracy')
plot1[0].plot(LossData, label='Train Loss')
plot1[1].plot(AccuracyData, label='Accuracy')

#plt.xlabel('Sample No')
#plot1[1].xlabel('Sample No')
#plot1[0].ylabel('Loss')
#plot1[1].ylabel('Accuracy')
plt.legend()
plt.show()
model.save(model_dir, overwrite=True)
