from matplotlib import pyplot as plt
import pickle

FileHandle = open('/Users/baherkherbek/Desktop/pyt/model_1/Results/Training Data', 'rb')
ModelPerformace = pickle.load(FileHandle)
FileHandle.close()

epoch = ModelPerformace['Epoch']
TrainingLoss = ModelPerformace['TrainingLoss']
Accuracy = ModelPerformace['Accuracy']


plt.subplot(1, 2, 1)
plt.title('Loss Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(list(range(0, max(epoch)+2, 2)))
plt.plot(epoch, TrainingLoss, 'go-')
plt.legend(['Training Loss'])

plt.subplot(1, 2, 2)
plt.title('Accuracy Per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy[%]')
plt.xticks(list(range(0, max(epoch)+2, 2)))
plt.yticks(list(range(0, 105, 5)))
plt.plot(epoch, Accuracy, 'ro-')
plt.legend(['Accuracy'], loc='upper left')

plt.show()
