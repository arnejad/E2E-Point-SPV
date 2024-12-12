######### GLOBAL PARAMS
FEAT_LAYERS_N = 6       # The layer to compare perception loss within ResNet
BATCH_SIZE = 16
LAMBDA = 0.75            # lambda * spatial loss + (1-lambda) * CE loss
LEARNING_RATE = 0.0001
EPOCHS = 100
LOG_ON_BATCH = 200      # logging the loss values on each x batches
DATASET = "COCO"
IMG_SIZE = 256          # images are resized to this after being read
SIMULATOR = "biological"       # training and validating with a chosen simulator "regular", "biological", "interpol"
EXEC_UNIT = "GPU"       # Computation unit. if "GPU" is chosen, the models and weights are created in GPU memory

# target classes from COCO. if this this left empty, the dataset returns all images that some might not include any segmentations 
#and causes execution failure. Limited number for test,
FILTERED_CATS = [ 'YOUR OBJECT CATEGORIES COME HERE, FOR INSTANCE:','person', 'dog', 'bicycle']  

NUM_OF_CLASSES = len(FILTERED_CATS)
FOVEA_SIZE = 80     # size of extracted gaze patch for central part of the vision 


# OUTPUT DIRECTORIES
LOG_DIR = 'res/log_bio.txt'
FIGS_DIR = 'res/figs_bio'
MODELS_DIR = 'res/models_bio'