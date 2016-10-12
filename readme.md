# ubam config

## windows 7 config

install git  
install anaconda3  
open command line (*windows*+*r*, type *cmd*, then *enter*)
```cmd
mode 190,50
git clone https://github.com/wangbx66/ubam
pip install theano
pip install lasagne
pip install h5py
pip install pulp
pip install gym
```
close the command line and open a new command line, then test with the following and ensure no error warnings
```cmd
python
import scipy
import theano
```

new a file *.theanorc* to *c:\users\szheng*, and write
```
#!sh
[global]
device=cpu
floatX=float32

[nvcc]
compiler_bindir=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin
flags=-LC:\Users\szheng\AppData\Local\Continuum\Anaconda3
```

## To Start With

First download datasets from Game Trace Archive, and supplementary datasets from kaggle. Remove all headers and non-ascii characters (keep escaped unicodes)

##Sam

UBAM DL


Step1: 


Agent.py % Parse and clean WoWH dataset
% Step1: Trajs
% Step2: Compress Trajs into HDF5



- reward() % calculate satisfaction, output data/trajs

- hdf_dump() % experience reply data



Architecure.py

- Buid and train Q model 

Example: 


python architecture.py 0 3000 300 0.0025 sam-sep-30

0 (reward index 0 means all 5f, 1 means f1 (advancement))

3000 ( batch x 24(frame/reply))

300 (loop length/horzion)

0.0025 (learning rate)

sam-sep-30 (file name for Q network)

Output file: 
Q Network
architecture.log (for training evaluation)


Recover.py 

% Load Q network
% Formulate iRL cons_gen function
% q_val_eval Q(st,a)
% pulpsol (Python Linear Programming Solver)




