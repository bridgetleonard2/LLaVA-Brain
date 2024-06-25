import fun
import numpy as np
import torch

class EncodingModel:
    def __init__(self, model, subject, layer):
        self.model_name = "llava"
        self.subject = subject
        self.layer = layer
        self.encoding_model = None
        self.correlations = None
        if self.model == "llava":
            self.layer_name = layer
        else:
            self.layer_name = f"layer_{layer}"
    
    def setup_model(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model_name == "llava":
            self.model, self.processor, self.features, self.layer_selected = fun.setup_llava(self.device,
                                                                                             self.layer)
        else:
            self.model, self.processor, self.features, self.layer_selected = fun.setup_bridgetower(self.device,
                                                                                             self.layer)
    
    def crossmodal_vision_model(self):
        print("Building vision model")
        self.encoding_model = fun.crossmodal_vision_model(self.model,
                                                          self.subject,
                                                          self.layer,
                                                          self.layer_name)
        print("Predicting fMRI data and calculating correlations")
        self.correlations = fun.story_prediction(self.model, self.subject,
                                                 self.layer,
                                                 self.layer_name,
                                                 self.encoding_model)
        np.save(f'results/movie_to_story/{self.subject}/' +
                f'layer{str(self.layer)}_correlations.npy',
                self.correlations)

    def crossmodal_language_model(self):
        print("Building language model")
        self.encoding_model = fun.crossmodal_language_model(self.subject,
                                                            self.layer)
        print("Predicting fMRI data and calculating correlations")
        self.correlations = fun.movie_prediction(self.subject, self.layer,
                                                 self.encoding_model)

    def withinmodal_vision_model(self):
        self.correlations = fun.withinmodal_vision_model(self.subject,
                                                         self.layer)

    def faceLand_vision_model(self, modality):
        self.encoding_model = fun.crossmodal_vision_model(self.subject,
                                                          self.layer)
        self.correlations = fun.faceLandscape_prediction(self.subject,
                                                         modality,
                                                         self.layer,
                                                         self.encoding_model)


if __name__ == "__main__":
    crossmodalVision = EncodingModel("S1", 4).crossmodal_vision_model()
    crossmodalLanguage = EncodingModel("S1", 4).crossmodal_language_model()
    withinmodalVision = EncodingModel("S1", 4).withinmodal_vision_model()
    faceVision = EncodingModel("S1", 4).faceLand_vision_model("face")
    landscapeVision = EncodingModel("S1", 4).faceLand_vision_model("landscape")
    print("Encoding models completed")
