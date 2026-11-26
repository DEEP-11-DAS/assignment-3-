# Q2: criterion and predictions (commit this file BEFORE running q2.py; edit to your own beliefs first)

## (a) Fairness criterion (defined before looking at Q2 results; models reused from Q1(a), where first introduced)
MLP (1,786,186 params) vs CNN (1,793,210 params): parameter counts within 1%; identical optimiser (AdamW),
weight decay, batch size, epochs (60), augmentation, normalisation, BatchNorm use; learning rate tuned
on the validation set from the same 5-value grid for each model; 3 seeds; checkpoint chosen by val acc.
Proposed reason for the gap: the CNN's locality + weight sharing exploit the spatial arrangement of pixels
(neighbouring pixels are correlated and the same feature is useful at many positions); the MLP has to learn
every position separately and treats pixel positions as unrelated.

## (b) Spatial disturbance: one fixed random permutation of the 1024 pixel positions (same for all channels)
Prediction, if explanation (a) is right:
- MLP: essentially unchanged (|change| < 1.5 points). A dense first layer has no notion of position, so a
  fixed permutation only relabels its input weights.
- CNN: large drop (>= 20 points), to around or below the MLP, because 3x3 windows now contain unrelated pixels.
If the CNN stays far above the MLP on permuted data, explanation (a) is rejected (its advantage would come from
depth, BatchNorm, pooling... not from spatial structure).

## (c) Two explanations for the trained CNN's behaviour
H_colour: it mostly relies on colour / local texture statistics.
H_shape : it mostly relies on spatial arrangement / global shape.
Interventions at test time on the Q1(a) CNN: grayscale; random block shuffle (2x2, 4x4, 8x8 grids: local
content kept, global layout destroyed); centre occluded; only centre kept.
Control: a CNN TRAINED on grayscale (separates "colour is needed" from "model is brittle to a shift").
Predictions: grayscale drops accuracy by a moderate amount (3-10 points) and the grayscale-trained CNN recovers most
of it (colour helps but is not essential). Block shuffle drops accuracy sharply and monotonically with grid size
(2x2 >= 15 points, 8x8 close to chance-like collapse), so shape/layout dominates. Wrong predictions under shuffle
are made with high confidence (overconfidence), concentrated on a few classes.
