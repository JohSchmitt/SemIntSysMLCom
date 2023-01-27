Seminar Intelligent Systems Machine Learning Competition WiSe 22  
University of Siegen  
Johannes Schmitt

# Requirements
autogluon  
	
scikit-learn  
imblearn  
pandas  
numpy  

# General Usage
Predictions will be saved in '_predictions'

Best predictions were achived with autogluon
- classification
	- clas_autogluon.py
- regression
	- regr_autogluon.py


Other experiments are saved in folder 'other'
- classification
	- average filter for feature 0 + autogluon
- regression


## AutoGluon (python 3.9)
1. run 'python3.9 clas_autogluon.py' or 'python3.9 regr_autogluon.py'

## Other
1. move file from folder 'other' to project root and run it
2. ..
3. profit
	
# Used System
## Hardware
- AMD Ryzen 7 2700X
- 48 GB RAM
- NVIDIA GeForce GTX 970

## AutoGluon
- Windows Subsystem for Linux (WSL) V2
- Ubuntu 22.04.1 LTS
- Python 3.9
- Pip for python 3.9

## Other
- Windows 10 Build 19045.2546
- Python 3.11
