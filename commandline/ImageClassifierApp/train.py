import os
from datetime import datetime
import TrainingHelper as th
import ClassifierHelper as ch
import ArgsParser as ap

try:
    args = ap.get_train_input_args()

    th.validate_data_directory(args.data_directory)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if (args.arch != "resnet") and (args.arch != "densenet"):
        raise ValueError("Invalid archectecture, must be resnet or densenet")            

    device = ch.get_device(args.gpu)
    dataloaders = th.get_data_loaders(args.data_directory,args.batch_size)
    model , criterion , optimizer  = th.create_model(args.arch,args.hidden_units,args.drop_out,args.learning_rate)
   # print(model)
    
    model_name = "model_" + args.arch + "_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".pth"
    print(f"Training model -> {model_name}")    
    th.train_model(model,criterion,optimizer,device,dataloaders,args.batch_size,args.epochs)
    
    th.save_model(model,model_name,args.save_dir,dataloaders['train'].dataset.class_to_idx)        
    #model.class_to_idx = dataloaders['train'].dataset.class_to_idx    
    #torch.save(model, args.save_dir + os.sep + model_name)
    #print(f"\nModel saved as {args.save_dir + os.sep + model_name}")
    
except Exception as e:
    print("An exception has occured")
    print(e)