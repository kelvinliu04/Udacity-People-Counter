import cv2
from openvino.inference_engine import IENetwork, IECore

class OpenVinoNetwork:
    '''
    OPENVINO INFERENCE TEMPLETE 
    - kelvinliu04@gmail.com 
    ---------------------------
    example using:
        person_det = Network('person-detection-retail-0013', folder=None)
        
        # ======================== for asynchronous
        person_det.async_inference(frame)
        if person_det.wait() == 0:
            person_det.get_output()
            
        # ======================== for synchronous
        person_det.sync_inference(frame)
        person_det.get_output()
    
    '''
    def __init__(self , model, folder='', device='CPU', cpu_extension=None): 
        model_bin = folder + "/" + model + ".bin"
        model_xml = folder + "/" + model + ".xml"
        # Load Plugin
        self.plugin = IECore()
        if cpu_extension and "CPU" in device:                # Add a CPU extension, if applicable
            self.plugin.add_extension(cpu_extension, device)
        # Load Model
        self.load_model(model_bin, model_xml, device)
        print('Model: {}'.format(model))

    def load_model(self, model_bin, model_xml, device):
        '''
        1) Load model only once
        '''
        # Load Model
        net = IENetwork(model=model_xml, weights=model_bin)
        self.exec_net = self.plugin.load_network(net, device)
        # Input Output Blob
        self.input_blob = next(iter(net.inputs))
        self.output_blob = next(iter(net.outputs))
        # Input Shape [BxCxHxW] B=N
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        # print Model Input Output 
        print('Input: {}'.format(self.input_blob))
        print('Output: {}'.format(self.output_blob))
        
    def preprocessing(self, image):
        '''
        2) prepare input (reshape img)
        '''
        img = cv2.dnn.blobFromImage(image, size=(self.w, self.h))
        return img
    
    def sync_inference(self, image):
        '''
        3.a)Using synchronous inference
        '''
        image = self.preprocessing(image)
        self.exec_net.requests[0].infer({self.input_blob: image})
        

    def async_inference(self, image):
        '''
        3.b.1) Makes an asynchronous inference request, given an input image.
        '''
        image = self.preprocessing(image)
        self.exec_net.start_async(request_id=0, 
            inputs={self.input_blob: image})
    
    def wait(self):
       '''
       3.b.2) Checks the status of the inference request.
       '''
       status = self.exec_net.requests[0].wait(-1)
       return status
   
    def get_output(self):
        '''
        4) Returns a list of the results for the output layer of the network.
        '''
        return self.exec_net.requests[0].outputs[self.output_blob]
    