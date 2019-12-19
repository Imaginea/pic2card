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

### create label map 
Edit training/object-detection.pbxt file to match the label maps mentioned in generate_tfrecord.py

### download the model(faster_rcnn_inception_v2_coco_2018_01_28) from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
download faster_rcnn_inception_v2_coco_2018_01_28 model and place it under object_detection/training dir  

### set below paths appropriately in pipeline.config file

    fine_tune_checkpoint ---- path to faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt
    tf_record_input_reader 
               {
                    input_path: ---- full path to "tf_records/train.record"
              }
              label_map_path: ---- full path to "training/object-detection.pbtxt"
                }
                
    eval_input_reader: 
               {
              tf_record_input_reader {
                input_path: ---- full path to "tf_records/test.record"
              }
              label_map_path: ---- full path to "/training/object-detection.pbtxt"
              shuffle: false
              num_readers: 1
            }
    
### train model using below command 

    python model_main.py --logtostderr --model_dir=training/ --pipeline_config_path=training/pipeline.config
    
### export inference graph 
After the model is trained, we can use it for prediction using inference graphs
change XXXX to represent the highest number of trained model 

    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/pipeline.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory ../inference_graph

### predict 
Jupyter notebook available under notebooks can be used for prediction
