# DeepBP
Realization of the paper
## Require
This project uses the following dependencies:
```
Python 3.9.18
fair-esm 2.0.0
pandas 1.3.5
numpy 1.21.6
scikit-learn 1.0.2
torch 1.13.0+cu116
tensorflow 2.6.0
keras 2.6.0
```
If other libraries are missing, you can install them yourself using the command `pip install package_name==2.0.0`
## ROC and PR graph
### ACE inhibitory peptides
![ACE inhibitory peptides](https://github.com/Zhou-Jianren/bioactive-peptides/blob/main/image/ACE.jpeg)
### ACP
![ACP](https://github.com/Zhou-Jianren/bioactive-peptides/blob/main/image/ACP.jpeg)
### Performance Comparison
![Performance Comparison](https://github.com/Zhou-Jianren/bioactive-peptides/blob/main/image/polar.jpeg)
## Usage Process
Note: The ACE inhibitory peptide dataset uses 0 and 1 to represent high activity and low/no activity, respectively.
In the ACP dataset, 0 represents ACP and 1 represents non-ACP
### Extract features
Feature extraction using ESM-2
### Model predictions
The model experimental methods are in the ACE and ACP folders respectively. The main folder contains command line predictions for ACE inhibitory peptides and ACP.
## Further adjustments and modifications
Feel free to make your own changes. Simply scroll down to the Model Architecture section and modify it to suit your needs.
If you have any questions, please contact us: 2521282343@qq.com.
