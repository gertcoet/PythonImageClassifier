
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import os
import sys
import numpy as np
import random
import argparse
import json


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model             
    width, height = image.size
    min_side_len = 256
    crop_size = 244
    
    # Create a thumbnail with the shortest side equal to min_side_len    
    if width > height:
        image.thumbnail((sys.maxsize, min_side_len),Image.ANTIALIAS) 
    else:  
        image.thumbnail((min_side_len, sys.maxsize),Image.ANTIALIAS)  
                
    # Crop the image to crop_size
    width, height = image.size
    crop_box = get_crop_box(width,height,crop_size)        
    image = image.crop(crop_box)
        
    # Convert the image to a numpy array and convert the colour channel position    
    np_image = np.array(image) / 255.0     
    
    # Normalize the image as per the model requirements
    mean = np.array([0.485, 0.456, 0.406])
    st_dev = np.array([0.229, 0.224, 0.225])    
    np_image = (np_image - mean) / st_dev     
    
    # Move color channel to first dimension
    np_image = np_image.transpose((2,0,1))            
    image_tensor = torch.from_numpy(np_image)  
    float_tensor = image_tensor.float()
    
    return float_tensor

# Used in process_image() to get a crop box
def get_crop_box(width,height,crop_to):                    
           
    left = (width - 224) / 2
    upper = (height - 224) / 2
    right = (width + 224) / 2
    lower = (height + 224) / 2
     
    return (left,upper,right,lower)    

# Gets the class of the image from the filepath
def get_class_from_path(path):
    path = path[::-1]
    pic_class = path.split(os.sep)
    return pic_class[1][::-1]

#Returns a random image path from the specified list
def get_random_image_path(image_dir):
    #Get a list of all the images
    file_paths = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            #if file.endswith(".jpg"):
            file_paths.append(os.path.join(root, file))
                
    #Check that a file was found
    if (len(file_paths) < 1):
        print("No input file found")
    else:
    #Pick a random file to display    
        file_path = file_paths[random.randint(0,len(file_paths))]
    
    return file_path    

def predict(image_path, model,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file   
    im = Image.open(image_path)
    im_tensor = process_image(im)
    im_tensor.unsqueeze_(0)    
    im_tensor = im_tensor.to(device)
    
    model.to(device)
    model.eval()

    pred_top_class = []
    pred_probs = []        
    
    with torch.no_grad():        
        pred_ps = torch.exp(model(im_tensor))          
        probs, pred_top_idx = pred_ps.topk(topk, dim=1)     
        
    pred_probs = probs[0].cpu().tolist()
    idx_list = pred_top_idx[0].cpu().numpy().astype(str).tolist()
    idx_to_class = {value: key for key, value in model.class_to_idx.items()}    
    
    try:
        for i in range(len(idx_list)):            
            pred_top_class.append(idx_to_class[int(idx_list[i])])                                
                     
    except:
        print(f"Failed to find class type {idx_list[i]}")
        pred_top_class = []
        pred_probs = []
        
    return pred_probs , pred_top_class    

def load_model(model_name,device_type):    
    if (device_type == 'cpu'):
        model = torch.load(model_name, map_location=lambda storage, loc: storage)
    else:    
        model = torch.load(model_name)        
    return model  

def get_class_names(class_list,catagory_dict):
    class_name = []
    for c in class_list:
        class_name.append(catagory_dict[str(c)])
    return class_name   

def get_device(gpu):
    if (gpu):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
    else:
        device = torch.device('cpu')    
    print(f"*** Using device type {device.type}  ***\n")    
    
    return device

def get_cat_dict(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)    
    return cat_to_name

