import functions
import numpy as np


class VisionEncodingModel:
    def __init__(self):
        self.Xtrain = None
        self.model = None
        self.predictions = None

        # Parameters
        self.layer = 'multi_modal_projector.linear_2'
        self.subject = 'S1'
        self.modality = 'face'

    def load_data(self):
        self.Xtrain, self.feature_arrays = functions.get_Xdata(self.layer)

    def train_model(self):
        Ytrain = functions.get_Ydata(self.subject)
        self.model = functions.run_model(self.Xtrain, Ytrain,
                                         self.feature_arrays)

    def predict(self):
        self.testData = functions.get_test_data(self.layer, self.modality)
        self.predictions = np.dot(self.testData, self.model)


model = VisionEncodingModel()
model.load_data()
model.train_model()
model.predict()

# Save predictions
np.save('results/' + model.modality + '/' + model.subject + '/' + model.layer +
        '_predictions.npy', model.predictions)
