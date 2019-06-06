import ClassifierHelper as ch
import ArgsParser as ap

try:
    #Get input parameter for Predict.py
    args = ap.get_pred_input_args()

    #Get the devicetype
    device = ch.get_device(args.gpu)

    #Get class mapper
    cat_to_name = ch.get_cat_dict(args.category_names)

    #Run image through the model
    model = ch.load_model(args.checkpoint,device.type)
    probs , top_k_class = ch.predict(args.input,model,device,args.top_k)
    actul_flower = cat_to_name[ch.get_class_from_path(args.input)]
    
    #Print results
    print("{:<15} *** {:s} ***\n".format("Input flower ",actul_flower))
    for i in range(len(top_k_class)):    
        print("{:<20} :  {:7.3f} %".format(cat_to_name[str(top_k_class[i])],100*probs[i]))
    print()
except Exception as e:
    print("There was an errro while excuting the program")
    print(e)    
