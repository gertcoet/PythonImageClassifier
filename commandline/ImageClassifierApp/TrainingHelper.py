import torch
from torch import nn ,optim
from torchvision import datasets, transforms, models
import numpy as np
import os
import datetime

# Check if the supplied folder contains required folders
def validate_data_directory(data_folder):
    test_folder,training_folder,validation_folder= False,False,False

    for dirs, folders, files in os.walk(data_folder):     
        for folder in folders:   
            if (folder == "test"):
                test_folder = True
            if (folder == "train"):
                training_folder = True                
            if (folder == "valid"):
                validation_folder = True      

    if (test_folder == False) or (training_folder == False) or (validation_folder == False):    
        raise Exception("The input folder does not cointain the required sub-folders")                        

def get_data_loaders(data_folder,batch_size): 
    train_dir = data_folder + os.sep + 'train'
    valid_dir = data_folder + os.sep + 'valid'
    test_dir = data_folder + os.sep + 'test'

    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])        

    test_data_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])      

    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=test_data_transforms)

    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size , shuffle=True)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size , shuffle=True )
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=batch_size , shuffle=True )

    dataloaders = {'train' : train_dataloaders,'test' : test_dataloaders, 'valid' : valid_dataloaders}

    return dataloaders

# Create a feed forward network for the classifier
def create_ff_Network(input_unit,output_units,hidden_units,drop_out):     
    if ((drop_out > 1.0) or (drop_out < 0.0)):
        raise ValueError("Dropout must be between 0 and 1")

    ff_network = nn.Sequential(nn.Linear(input_unit, hidden_units),nn.ReLU(),nn.Dropout(drop_out),nn.Linear(hidden_units, output_units),nn.LogSoftmax(dim=1))

    return ff_network

# Create a new model
def create_model(architecture,hidden_units,drop_out,learning_rate):
    if (architecture == "resnet"):
        model = models.resnet18(pretrained=True)    
        #freez paramaters
        for param in model.parameters():
            param.requires_grad = False    
        model.fc = create_ff_Network(512,102,hidden_units,drop_out)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        model = models.densenet121(pretrained=True)
        #freez paramaters
        for param in model.parameters():
            param.requires_grad = False    
        model.classifier = create_ff_Network(1024,102,hidden_units,drop_out)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)         
    
    #Set loss function 
    criterion = nn.NLLLoss()    

    return model , criterion , optimizer    
    
def train_model(model,criterion,optimizer,device,dataloaders,batch_size=64,epochs=3):
        
    run_number = 0
    running_loss = 0    

    model.to(device);   
    for epoch in range(epochs):   
        start = datetime.datetime.now()
        
        for images, labels in dataloaders['train']:
            
            images = images.to(device)             
            labels = labels.to(device)            
            
            run_number += 1                                         
            optimizer.zero_grad()

            logps = model.forward(images)        
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if run_number % batch_size == 0:
                
                test_loss = 0
                accuracy = 0
                
                model.eval()
                with torch.no_grad():
                    for test_images, test_labels in dataloaders['test']:
                                                
                        test_images = test_images.to(device) 
                        test_labels = test_labels.to(device)
                        
                        # Calculate batch loss
                        test_logps = model.forward(test_images)                    
                        batch_loss = criterion(test_logps, test_labels)
                        test_loss += batch_loss.item()

                        # Calculate batch accuracy
                        ps = torch.exp(test_logps)                    
                        prob, pred_class = ps.topk(1, dim=1)
                        equals = pred_class == test_labels.view(*pred_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1} of {epochs} | " \
                      f"Training loss -> {running_loss/batch_size:.3f} | " \
                      f"Test loss -> {test_loss/len(dataloaders['test']):.3f} | " \
                      f"Test accuracy -> {accuracy/len(dataloaders['test']):.3f}") \
                
                running_loss = 0
                model.train()    

        print (f"Epoch run time -- {(datetime.datetime.now() - start)}")             
        
def save_model(model,model_save_name,save_dir,class_to_index):        
    model.class_to_idx = class_to_index
    torch.save(model, save_dir + os.sep + model_save_name)
    print(f"\nModel saved as --> {save_dir + os.sep + model_save_name}")
                

              
     
