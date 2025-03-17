# Wandb report link
https://api.wandb.ai/links/arunangshudutta218-iitm/fjeyu4w5

# To run train.py

- Do ```python train.py --wandb_entity ENTITY_NAME --wandb_project PROJECT_NAME``` to run the script, where ```ENTITY_NAME``` & ```PROJECT_NAME``` is your entity name and proejct name. Currently, the default is set to mine.

- ```train.py``` can handle different arguments. The defualts are set to the hyperparameters which gave me the best validation accuracy.
     
 Arguments supported are:
     
| Name | Default Value | Description |
| --- | ------------- | ----------- |
| `-wp`, `--wandb_project` | name | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | name | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mse", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-beta`, `--beta` | 0.5 | Beta used by momentum, nag optimizers and rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-w_d`, `--weight_decay` | 0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "relu"] |

```train.py``` returns the wandb logs generated on Training and Validation dataset. Also, The Real train and test accuracy are printed out.

An example, if you want to train the model on ```mnist dataset``` with same configurations.

Run the following command: ```python train.py --wandb_entity ENTITY_NAME --wandb_project PROJECT_NAME -d mnist```
