## Comparing voxelwise encoding models using different multimodal models

### Vision only voxelwise encoding model
How well does a model trained on model features from movie stimuli and brain data from movie stimuli predict brain responses?

<div align="center">
  <img src="/bridgetower_comparisons/layer8_visual.png" alt="BridgeTower" style="width:45%; float: left; margin-right: 2%;" />
  <img src="/results/multi_modal_projector.linear_2/evalcorr.png" alt="LLaVA" style="width:52%; float: left;" />
  <br>
  <i>Using the vision encoding model to predict fMRI responses to movies (leave one out train/eval method). BridgeTower (left) vs LLaVA result (right)</i>
</div>

### Vision voxelwise encoding model with face images
Does the model capture well known phenomena (generalize) like the activation of FFA in response to images of faces?

<div align="center">
  <img src="/bridgetower_comparisons/layer8_face.png" alt="BridgeTower" style="width:45%; float: left; margin-right: 2%;" />
  <img src="/results/multi_modal_projector.linear_2/pred_face.png" alt="LLaVA" style="width:52%; float: left;" />
  <br>
  <i>Using the vision encoding model to predict fMRI responses to faces. BridgeTower (left) vs LLaVA result (right)</i>
</div>