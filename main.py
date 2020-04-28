"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file, 'CAM' if using webcam")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser

def post_detection(result, frameShape, confident=0.5):
    '''
    this is my additional function for extract the result
    -kelvin liusiani
    '''
    boxes = []
    scores = []
    for detection in result[0][0]:
        if detection[2]> confident:
            img_h, img_w, _ = frameShape
            xmin = int(detection[3] * img_w)
            ymin = int(detection[4] * img_h)
            xmax = int(detection[5] * img_w)
            ymax = int(detection[6] * img_h)
            box = xmin, ymin, xmax, ymax
            boxes.append(box)
            scores.append(detection[2]) 
    return boxes, scores

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###(ok)
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # my) init parameters 
    current_count = 0 
    total_count = 0 
    duration = 0 
    last_count = 0 
    start_time = 0
    isFirst = True
    single_image_mode = False
    
    # Initialise the class (ok)
    infer_network = Network()
    
    # Set Probability threshold for detections (ok)
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ### (ok)
    infer_network.load_model(args.model, device ="CPU", cpu_extension=args.cpu_extension)
    n, c, h, w = infer_network.get_input_shape()

    ### TODO: Handle the input stream ### (ok)
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_image_mode = True
        input_stream = args.input
    else:
        input_stream = args.input
        #assert os.path.isfile(args.input), "Specified input file doesn't exist"
    
    cap = cv2.VideoCapture(input_stream)
    cap.open(input_stream)

    ### TODO: Loop until stream is over ###(ok)
    while cap.isOpened():
        ### TODO: Read from the video capture ###(ok)
        ret, frame = cap.read()
        key_pressed = cv2.waitKey(60)
        if not ret:
            break
    
        ### TODO: Pre-process the image as needed ###(ok)
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        

        ### TODO: Start asynchronous inference for specified request ###(ok)
        infer_network.exec_net(image)

        ### TODO: Wait for the result ###(ok)
        if infer_network.wait() == 0:
            ### TODO: Get the results of the inference request ###(ok)
            result = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###(ok)
            boxes, score = post_detection(result, frame.shape, prob_threshold)
            
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 238, 255), 1)
                
                
            
            ### TODO: Calculate and send relevant information on ###(ok)
            ### current_count, total_count and duration to the MQTT server ###
            if len(boxes) != current_count:
                if isFirst:
                    ts1 = time.time()
                    isFirst = False
                if time.time() - ts1 > 0.5:
                    current_count = len(boxes)
                    isFirst = True
                    
            
            ### Topic "person": keys of "count" and "total" ###(ok)
            if current_count > last_count:
                start_time = time.time()
                total_count = total_count + current_count - last_count
                client.publish("person", json.dumps({"total": total_count}))
            
            ### Topic "person/duration": key of "duration" ###(ok)
            if current_count < last_count:
                duration = int(time.time() - start_time)
                # Publish messages to the MQTT server
                client.publish("person/duration",
                               json.dumps({"duration": duration}))
            client.publish("person", json.dumps({"count": current_count}))   
            last_count = current_count
            if key_pressed == ord('q'):
                break

        ### TODO: Send the frame to the FFMPEG server ###(ok)
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###(ok)
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)
        
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    infer_network.clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
