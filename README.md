# Fea-DA---Uknown-SAR-Target-Identification-Method
This is a novel unknown sar target identification method based on feature extraction networks and KLD-RPA joint discrimination. Experiment results form MSTAR dataset shows that our proposed Fea-DA achieves state of the art unknown sar target identification accuracy while maintaining the high recognition accuracy of known target.

environment

python 3.7, pytorch 1.6 

First Step:
     implement the data2mat.py to transform the original images into .mat type.
Second Step:
     using the trian_FEN.py to trian the dataset, then use test_FEN.py to test, save the target features.
Third Step:
     launch KLD-RPA.py, this is an unknown sar target joint discrimination scheme to realize high accuracy identification of unknown sar target.

if this project could provide any help to you, please cite our paper-      .
thank you.
