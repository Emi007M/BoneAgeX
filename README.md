# Bone Age Assessment System

## Download
image packages under tmp files available on email

## Installation
runs under  Python 3.6.2
```
git clone https://github.com/Emi007M/BoneAgeX.git
cd BoneAgeX
python3.6 -m pip install --upgrade pip
python3.6 -m pip install --upgrade setuptools
pip install .
```

## Launching:

Help
python3.6 main.py --help

Shortest command should get input images from -i, create net model -n and use it to predict 10 images from the dataset -z
```
python3.6 main.py -i "C:/.../BoneAge/imgs_sm" -nz
```

Training
1. Get input images from -i,
2. save model during training to -m,
3. train model (-t) for -e epochs with batch -b and batch for evaluations -v
4. using -g gpu (amount)
5. create new model -n
```
python3.6 main.py -i "C:/.../BoneAge/imgs_sm" -m "trained_models/" -t -e 20 -b 16 -v 32 -g 1 -n
```

Separate evaluation
1. Get input images from -i,
2. load model from -l,
3. evaluate input sets (-o),
4. test predictions on small subset (-z)
```
python3.6 main.py -i "C:/.../BoneAge/imgs_sm" -l "trained_models/5/"  -oz
```



