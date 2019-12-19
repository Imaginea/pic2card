# Mystique

This repo contains code for converting adaptive cards GUI design image into Json 
faster-rcnn neural model is used for this purpose

#Environment Setup:

This repo implements tensorflow research models, hence it necessary to have that repo in your machine 
    
    https://github.com/tensorflow/models.git

extend PYTHONPATH with newly cloned repo
    
    cd models/models/reseach/
    export PYTHONPATH=$PYTHONPATH:`pwd`
    export PYTHONPATH=$PYTHONPATH:`pwd`/slim

# Steps for Object detection
### create csv files for train and test datasets

    cd utils
    python xml_to_csv.py

### set configs for generating tf records
Edit utils/generate_tfrecord.py file to meet the object labels
example : 
    
    # TO-DO replace this with label map
    def class_text_to_int(row_label):
        if row_label == 'textbox':
            return 1
        if row_label == 'radio_button':
            return 2
        if row_label == 'checkbox':
            return 3
        else:
            None

### generate tf records for train and test sets

    python generate_tfrecord.py --csv_input=../images/train_labels.csv --image_dir=../images/train --output_path=../tf_records/train.record
    python generate_tfrecord.py --csv_input=../images/test_labels.csv --image_dir=../images/test --output_path=../tf_records/test.record



    
 
