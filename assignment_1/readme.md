# Wandb report link
https://api.wandb.ai/links/arunangshudutta218-iitm/fjeyu4w5

# To run train.py

- Do ```python train.py --wandb_entity ENTITY_NAME --wandb_project PROJECT_NAME``` to run the script, where ```ENTITY_NAME``` & ```PROJECT_NAME``` is your entity name and proejct name. Currently, the default is set to mine.

- ```train.py``` can handle different arguments. The defualts are set to the hyperparameters which gave me the best validation accuracy.
     
 Arguments supported are:
     
| Name | Default Value | Description |
| --- | ------------- | ----------- |
| `-wp`, `--wandb_project` | Assignment-1 | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | shashwat_mm19b053  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 30 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 32 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mse", "cross_entropy"] |
| `-o`, `--optimizer` | adam | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.9 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.9 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.9 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.999 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.0000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | 0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | Xavier | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 3 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 128 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | tanh | choices:  ["identity", "sigmoid", "tanh", "relu"] |

```train.py``` returns the wandb logs generated on Training and Validation dataset. Also, The Real train and test accuracy are printed out.

An example, if you want to train the model on ```mnist dataset``` with same configurations.

Run the following command: ```python train.py --wandb_entity ENTITY_NAME --wandb_project PROJECT_NAME -d mnist```
