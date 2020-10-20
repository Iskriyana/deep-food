"""Some helper functions for building the app
"""

import sys
sys.path.append('..')
from food_identification.models.model_helpers import object_detection_sliding_window

# ENABLE FOR LOCAL PREDICTIONS:
#
#from tensorflow.keras.models import load_model
#from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception_v3
# 
#def load_final_model():
#    loaded_model = load_model(
#        MODEL_PATH,
#        custom_objects=None,
#        compile=True
#    )
#    return loaded_model



def make_prediction(image, model, ind2class, sliding_strides=[64]):
    """Run prediction pipeline (CNN + Sliding Window)
    
    Args:
        image: Input image
        model: Model specs or loaded model (for local predictions)
        ind2class: Dict mapping from integers to labels
        sliding_strides: (optional) Optimal stride was [64] but larger values
            can speed up the model significantly
        
    Returns:
        pred_labels: predictions
        probabilities: probabilities
        x0: x-coordinates
        y0: y-coordinates
        windowsize: box sizes
    """
    pred_labels, probabilities, x0, y0, windowsize = object_detection_sliding_window(
        model=model, 
        input_img=image, 
        preprocess_function=lambda x: x,
        kernel_size=224, 
        ind2class=ind2class, 
        scaling_factors=[1.5], 
        sliding_strides=sliding_strides, 
        thr=0.95, 
        overlap_thr=0.2)
    
    return pred_labels, probabilities, x0, y0, windowsize