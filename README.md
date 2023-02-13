# Regularisierung von tiefen Neuronalen Netzen zur Bildgenerierung mittels xAI:Evaluierung der Effektivität von eXplainable LDGAN gegenüber State-of-the-Art Regularisierungsmethoden [Regularization of deep neural networks for image generation using xAI:Evaluation of the effectiveness of eXplainable LDGAN versus state-of-the-art regularization methods.Regularization of deep neural networks for image generation using xAI:Evaluation of the effectiveness of eXplainable LDGAN versus state-of-the-art regularization methods]

This repository contains the code moduls and data for our assignment with the same name.

### Data
- [CIFAR-10 Dataset ](https://www.cs.toronto.edu/~kriz/cifar.html)
### Main Module
- [Main to Start the experiment ](https://github.com/Kim-Kristin/regularization_xAI_LDGAN/blob/main/model/main.py)
### Requirements and Installation
To setup your environment, run :

For Linux or MacOS
```
setup script: `./setup.sh` or `sh setup.sh`

```
or for Windows
```
setup script `.\setup.ps1`
```

#### Troubleshooting: If this is not possible, please execute the following commands step by step in your command line from your development environment.
Then activate the python environment:

For Linux or MacOS

```
python -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For Windows
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```
After your environment it setup, run the main.py to start the experiment.

