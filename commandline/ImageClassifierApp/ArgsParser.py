import argparse

def get_train_input_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type = str, default = "/home/workspace/ImageClassifier/flowers", help = 'Directory containing images to process') 
    parser.add_argument('--save_dir', type = str, default = "/home/workspace/ImageClassifier/checkpoints", help = 'folder to save model checkpoints') 
    parser.add_argument('--gpu', type = bool, default = True, help = 'use a cude enabled device if available') 
    parser.add_argument('--arch', type = str, default = "resnet", help = 'network architecture to use (resnet or densenet)') 
    parser.add_argument('--hidden_units', type = int, default = 250, help = 'number of units in the hidden layers of the classifier') 
    parser.add_argument('--learning_rate', type = int, default = 0.003, help = 'the learning rate of the model') 
    parser.add_argument('--epochs', type = int, default = 3, help = 'number of epochs to run') 
    parser.add_argument('--drop_out', type = int, default = 0.2, help = 'drop out during training (must be between 0.0 and 1.0)') 
    parser.add_argument('--batch_size', type = int, default = 64, help = 'stochastic bacth size') 
    args = parser.parse_args() 
    return args

def get_pred_input_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type = str, default = "/home/workspace/ImageClassifier/flowers/valid/1/image_06739.jpg", help = 'path to a picture to be classified') 
    parser.add_argument('checkpoint', type = str, default = "/home/workspace/ImageClassifier/checkpoints/resnet_trained_flower.pth", help = 'path to a picture to be classified') 
    parser.add_argument('--gpu', type = bool, default = False, help = 'use a cude enabled device if available') 
    parser.add_argument('--top_k', type = int, default = 5, help = 'number of classes to return') 
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'json file containing catagory to name mappings') 
    args = parser.parse_args() 
    return args
