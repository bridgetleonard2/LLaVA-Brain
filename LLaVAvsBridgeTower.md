## Comparing voxelwise encoding models using different multimodal models

### Vision only voxelwise encoding model
How well does a model trained on model features from movie stimuli and brain data from movie stimuli predict brain responses?

<div align="center">
  <img src="/bridgetower_comparisons/layer8_visual.png" alt="BridgeTower" style="width:45%; float: left; margin-right: 2%;" />
  <img src="/results/multi-modal_projector/bridgetower_pipeline/vision_model.png" alt="LLaVA" style="width:45%; float: left;" />
  <br>
  <i>Using the vision encoding model to predict fMRI responses to movies (leave one out train/eval method). BridgeTower (left) vs LLaVA result (right)</i>
</div>

### Vision voxelwise encoding model with face images
Does the model capture well known phenomena (generalize) like the activation of FFA in response to images of faces?

<div align="center">
  <img src="/bridgetower_comparisons/layer8_face.png" alt="BridgeTower" style="width:45%; float: left; margin-right: 2%;" />
  <img src="/results/multi-modal_projector/bridgetower_pipeline/face.png" alt="LLaVA" style="width:40%; float: left;" />
  <br>
  <i>Using the vision encoding model to predict fMRI responses to faces. BridgeTower (left) vs LLaVA result (right)</i>
</div>


### Vision voxelwise encoding models face vs landscape images
Can we see FFA activation more clearly by subtracting out activation induced by landscape photos?

<div align="center">
  <img src="/bridgetower_comparisons/layer8_FaceMinusLandscape_top.png" alt="BridgeTower" style="width:45%; float: left; margin-right: 2%;" />
  <img src="/results/multi-modal_projector/bridgetower_pipeline/faceVSland.png" alt="LLaVA" style="width:40%; float: left;" />
  <br>
  <i>Using the vision encoding model to predict fMRI responses to faces. BridgeTower (left) vs LLaVA result (right)</i>
</div>


### Crossmodal prediction
How well do models trained on only visual stimuli do at predicting fMRI data for language stimuli and vice versa?

#### Movie to Story (vision model predicting story brain activity)
<div align="center">
  <img src="/bridgetower_comparisons/layer8_movie_to_story.png" alt="BridgeTower" style="width:45%; float: left; margin-right: 2%;" />
  <img src="/results/multi-modal_projector/bridgetower_pipeline/movie_to_story.png" alt="LLaVA" style="width:45%; float: left;" />
  <br>
  <i>Using the vision encoding model to predict fMRI responses to stories. BridgeTower (left) vs LLaVA result (right)</i>
</div>

#### Story to Movie (language model predicting movie brain activity)
<div align="center">
  <img src="/bridgetower_comparisons/layer8_story_to_movie.png" alt="BridgeTower" style="width:45%; float: left; margin-right: 2%;" />
  <img src="/results/multi-modal_projector/bridgetower_pipeline/story_to_movie.png" alt="LLaVA" style="width:45%; float: left;" />
  <br>
  <i>Using the language encoding model to predict fMRI responses to movies. BridgeTower (left) vs LLaVA result (right)</i>
</div>