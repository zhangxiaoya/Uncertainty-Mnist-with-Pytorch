# Uncertainty_Mnist
Uncertainty estimation on Mnist dataset

This is a PyTorch implementation of Dropout Uncertainty on Mnist. The experiment setting is based on [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf) at 5.2 Model Uncertainty in Classification Tasks.

## Installation 

1. Install Pytorch
[Pytorch](http://pytorch.org/)
[INSTALLING PREVIOUS VERSIONS OF PYTORCH](https://pytorch.org/get-started/previous-versions/)

Use virtualenv create pytorch env.
```
virtualenv --no-site-packages -p python3.6 ~/cenv_pytorch
source ~/venv_pytorch/bin/activate
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install glog
```

2. Clone this repository 
```
git clone https://github.com/andyhahaha/Uncertainty_Mnist
```

## Usage 

1. Train Lenet standard and Lenet dropout
```
python main.py --mode 0
```

2. Test Lenet standard and Lenet dropout
```
python main.py --mode 1
```

3. Test the Lenet dropout on rotated Mnist image 
```
python main.py --mode 2
```

## Result
These results show the uncertainty of different rotated digits.

<table>
  <tr>
    <td>0</td>
    <td>1</td>

  </tr>
  <tr>
    <td><img src="images/Digit0.png"/></td>
    <td><img src="images/Digit1.png"/></td>

  </tr>
  <tr>
    <td>2</td>
    <td>3</td>

  </tr>
  <tr>
    <td><img src="images/Digit2.png"/></td>
    <td><img src="images/Digit3.png"/></td>
  </tr>
  
  <tr>
    <td>4</td>
    <td>5</td>

  </tr>
  <tr>
    <td><img src="images/Digit4.png"/></td>
    <td><img src="images/Digit5.png"/></td>
  </tr>
  
  <tr>
    <td>6</td>
    <td>7</td>

  </tr>
  <tr>
    <td><img src="images/Digit6.png"/></td>
    <td><img src="images/Digit7.png"/></td>
  </tr>
  
  <tr>
    <td>8</td>
    <td>9</td>

  </tr>
  <tr>
    <td><img src="images/Digit8.png"/></td>
    <td><img src="images/Digit9.png"/></td>
  </tr>
 
</table>
