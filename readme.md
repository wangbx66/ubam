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
[global]
device=cpu
floatX=float32
cxx=
```
