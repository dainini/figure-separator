# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
author: satoshi tsutsui
Bulk figure extractor
'''

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from code.utils import postprocess, preprocess, load_graph
import os
import cv2
import argparse
import json

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--images", type=str, help=u"the directory that has figures", required=True)
parser.add_argument("--model", default="./data/figure-sepration-model-submitted-544.pb", type=str, help=u"model pb file. Default is ./data/figure-sepration-model-submitted-544.pb")
parser.add_argument("--thresh", default=0.5, type=float, help=u"sub-figuere detection threshold. Default is 0.5")
parser.add_argument("--output", default="./results", type=str, help=u"output directory ./results")
parser.add_argument("--annotate", default=0, type=int, help=u"save annotation to the image or not. 1 is yes, 0 is no. Default is 0.")
args = parser.parse_args()

# Network settings
meta = {'object_scale': 5, 'classes': 1, 'out_size': [17, 17, 30], 'colors': [(0, 0, 254)], 'thresh': args.thresh, 'anchors': [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52], 'num': 5, 'labels': ['figure']}

# Annotation settings
annotate = args.annotate == 1

# Load graph
graph = load_graph(args.model)

# List image files
images = os.listdir(args.images)

sub_figures = []
with tf.Session(graph=graph) as sess:
    print("---------------")
    print("Input directory: %s" % args.images)
    print("The directory has %s images" % len(images))
    print("Extraction started")
    for img_file in images:
        # Load image
        imgcv, imgcv_resized, img_input = preprocess(args.images + "/" + img_file)

        # Check if it is really an image or not
        if imgcv is None:
            print("%s is skipped because it is corrupted or not an image file." % img_file)
            continue
        else:
            print(img_file)

        # Detect it!
        detections = sess.run('output:0', feed_dict={'input:0': img_input})

        # Post-process it
        sub_figures, annotated_image = postprocess(meta, detections, imgcv, annotate)

        # If annotation is enabled, save the annotated image
        if annotate:
            annotated_image_name = os.path.join(args.output, img_file + ".annotated.png")
            cv2.imwrite(annotated_image_name, annotated_image)

        # Save output json
        json_name = os.path.join(args.output, img_file + ".json")
        with open(json_name, 'w') as f:
            json.dump(sub_figures, f, sort_keys=True, indent=4)
