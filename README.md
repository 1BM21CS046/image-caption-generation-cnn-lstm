# Image Caption Generation Using CNN–LSTM

Implementation of the CNN–LSTM image caption generation system described in:

**D. Prannav et al., “Image Caption Generation Using Deep Learning”  
Journal of Information Systems Engineering and Management, 2025**

## Architecture
- CNN Encoder: InceptionV3
- LSTM Decoder for caption generation

## Dataset
- Flickr8k

## Steps
1. Extract features  
   `python feature_extraction/extract_features.py`

2. Train model  
   `python model/train.py`

3. Generate captions  
   `python inference/caption_image.py`

## Note
Trained models and feature files are excluded due to size constraints.# image-caption-generation-cnn-lstm
