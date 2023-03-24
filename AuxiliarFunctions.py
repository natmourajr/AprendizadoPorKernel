import pickle


def write_list_of_hidden_neurons(filename, hidden_neurons):
    with open(filename,'wb') as file_handler:
        pickle.dump([hidden_neurons],file_handler)
    return 0
def get_list_of_hidden_neurons(filename):
    with open(filename,'rb') as file_handler:
        [hidden_neurons] = pickle.load(file_handler)
        
    return hidden_neurons

def update_model_status(filename, ifold, ineuron, iinit, status):
    with open(filename,'rb') as file_handler:
        [model_status] = pickle.load(file_handler)
    model_status[ifold, ineuron, iinit] = status
    with open(filename,'wb') as file_handler:
        pickle.dump([model_status],file_handler)
    return 0


def get_train_description(df_config, train_id):
    str_out = '\n\n\n'
    str_out +=  '=======================================\n'
    str_out +=  '%s Training Process'%(df_config['label'][train_id])
    str_out += '\n'
    str_out +=  '=======================================\n'
    
    str_out += 'Processing %s'%(df_config['train_data_path'][train_id])+'\n'
    str_out += 'Hidden Neurons:'
    hidden_neurons = ' '
    for ihidden_neuron in list(get_list_of_hidden_neurons(df_config['model_neurons'][train_id])):
        hidden_neurons += str(ihidden_neuron)+', '
    str_out += hidden_neurons[:-2]
    str_out += '\n'
    str_out += 'CV Folds: %s\n'%(df_config['cv_folds'][train_id])
    str_out += 'Inits: %s\n'%(df_config['model_inits'][train_id])
    return str_out