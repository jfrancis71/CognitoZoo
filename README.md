# Welcome to CognitoZoo

CognitoZoo implements a number of popular object detection models, including TinyYolo, SSDMobileNet, SSDVGG300,
and RetinaNet. The Single Shot Detector models are also in the Wolfram Neural Net repository (they were submitted by me and accepted into the repository). The other models are not yet at present.

## Installation

From the command line: 

git clone https://github.com/jfrancis71/CognitoZoo.git

This should be run from a working directory where you intend to install CognitoZoo.

## In Mathematica:

Note on first use of running CZDetectObjects.m it will pull all the neural models from my Wolfram repository. This may take a few minutes (depending on your internet connection speed). These models are then cached on your local machine.
The SetDirectory command below should be set to where you installed CognitoZoo.

![output](https://github.com/jfrancis71/CognitoZoo/blob/master/doc/images/CognitoDemo.jpg)

CZDetectObject and CZHighlightObjects supports AcceptanceThreshold and MaxOverlapFraction options.

Note the top level Mathematica packages are intended for general public use. You are welcome to use any packages in the subfolders, but they may contain platform specific code (and may not be useful without modification).

[Licensing Information](https://github.com/jfrancis71/CognitoZoo/Credits)
