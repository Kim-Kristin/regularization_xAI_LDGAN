# Regularisierung von tiefen Neuronalen Netzen zur Bildgenerierung mittels xAI:Evaluierung der Effektivität von eXplainable LDGAN gegenüber State-of-the-Art Regularisierungsmethoden   [Regularization of deep neural networks for image generation using xAI: Evaluation of the effectiveness of eXplainable LDGAN versus state-of-the-art regularization methods]

This repository contains the code moduls and data for our assignment with the same name.

### The objective of this repository
The objective of the implemented experiment is to evaluate the xAI-LDGAN as a regularization method against universal state-of-the-art regularization methods for Generative Adversarial Networks. The result of this evaluation is a comparison between the xAI-LDGAN and conventional regularization methods. This should provide information on whether the xAI-LDGAN is a better alternative to the universal state-of-the-art regularization methods.

#### Baseline Model
- DCGAN
#### State-of-the-art regularization methos
- Gradient Penalty
- Wasserstein GAN
- Wasserstein GAN with gradient Penalty
- Instance Normalization

#### New regularisation method
- xAI Limited Discriminator GAN

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

### References
- [xAIGAN ](https://github.com/explainable-gan/XAIGAN)
-

### Examples of generated Images of the Training for each method after 5 epochs
#### DCGAN
![DCGAN](https://github.com/Kim-Kristin/regularization_xAI_LDGAN/blob/main/outputs/VanillaGAN/gen_img-0005.png)
#### Gradient Penalty
![Gradient Penalty](https://github.com/Kim-Kristin/regularization_xAI_LDGAN/blob/main/outputs/GradientPenaltyGAN/gen_img-0005.png)
#### WGAN
![WGAN](https://github.com/Kim-Kristin/regularization_xAI_LDGAN/blob/main/outputs/WeightClippingGAN/gen_img-0005.png)
#### WGAN-GP
![WGAN-GP](https://github.com/Kim-Kristin/regularization_xAI_LDGAN/blob/main/outputs/WGANGP/gen_img-0005.png)
#### Instance Normalization
![Instance Normalization](https://github.com/Kim-Kristin/regularization_xAI_LDGAN/blob/main/outputs/NormalizationGAN/gen_img-0005.png)
#### xAI-LDGAN
![LDGAN](https://github.com/Kim-Kristin/regularization_xAI_LDGAN/blob/main/outputs/LDGAN/gen_img-0005.png)

