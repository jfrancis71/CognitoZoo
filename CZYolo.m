(* ::Package:: *)

(* Implements tiny YOLO on Mathematica version 11.

   tiny YOLO is a computer vision object detection and localisation model designed to detect
   20 object categories (e.g. people, horses, dogs etc)

   The code is based on the tiny YOLO model from Darknet, Joseph Redmon:
      https://pjreddie.com/darknet/yolo/
      
      Citation:
      @misc{darknet13,
      author =   {Joseph Redmon},
      title =    {Darknet: Open Source Neural Networks in C},
      howpublished = {\url{http://pjreddie.com/darknet/}},
      year = {2013--2016}
      }      

   See https://github.com/jfrancis71/CognitoZoo/wiki/Yolo
   for usage details
   
   Currently I don't have any non max suppression, so results can be a little cluttered.
   Takes about 25 secs to run on an image (MacBook air, CPU). This is actually quite slow
   by comparison with what can be achieved. (Darknet has reported tiny YOLO running at
   over 100 frames/sec).
   
   The codebase is fairly small, this is partly a consequence of quite tight coupling between the code
   and the weight file, so be careful to use the correct version of the weight file.
*)


(*

   The weight file used is the pretrained weight file from Darknet (http://pjreddie.com/darknet/)

   You will need to download the tiny YOLO weight file and install it on a Mathematica search path, eg your home directory.
   The below license notice (downloaded from Darknet on 02/06/2017) applies to the weight file indicated:

                                  YOLO LICENSE
                             Version 2, July 29 2016

THIS SOFTWARE LICENSE IS PROVIDED "ALL CAPS" SO THAT YOU KNOW IT IS SUPER
SERIOUS AND YOU DON'T MESS AROUND WITH COPYRIGHT LAW BECAUSE YOU WILL GET IN
TROUBLE HERE ARE SOME OTHER BUZZWORDS COMMONLY IN THESE THINGS WARRANTIES
LIABILITY CONTRACT TORT LIABLE CLAIMS RESTRICTION MERCHANTABILITY. NOW HERE'S
THE REAL LICENSE:

0. Darknet is public domain.
1. Do whatever you want with it.
2. Stop emailing me about it!


   Weight file available here (as of 27 Nov 2016):
      https://pjreddie.com/media/files/tiny-yolo-voc.weights
      
   In case it changes, I have an additional copy here:
      https://drive.google.com/file/d/0Bzhe0pgVZtNUeTNJN0w3OWRKOWM/view?usp=sharing
*)   


(* Copyright Julian Francis 2017. Please see license file for details. *)


<<CZUtils.m


(* resizes image so that length of longest side = max *)
CZMaxSideImage[image_Image, max_Integer] :=
 If[ImageAspectRatio[image] <1,
  ImageResize[image, max],
ImageResize[image,Scaled[max/ImageDimensions[image][[2]]]]
]


CZImagePadToSquare[image_Image]:=
   If[ImageAspectRatio[image]<1,
   ImagePad[image,{{0,0},{(1/2)*(ImageDimensions[image][[1]]-ImageDimensions[image][[2]]),Ceiling[(1/2)*(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])]}},Padding->0.5],
   ImagePad[image,{{(1/2)*(ImageDimensions[image][[2]]-ImageDimensions[image][[1]]),Ceiling[(1/2)*(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])]},{0,0}},Padding->0.5]]


file=OpenRead["tiny-yolo-voc.weights",BinaryFormat->True];


BinaryReadList[file,"Integer32",4]; (*Some magic numbers indicating file format versioning *)


ReadConvolutionLayer[file_,{outputLayers_,inputLayers_,w_,h_}]:=(
convb=BinaryReadList[file,"Real32",outputLayers];
convScales=BinaryReadList[file,"Real32",outputLayers];
convRM=BinaryReadList[file,"Real32",outputLayers];
convRV=BinaryReadList[file,"Real32",outputLayers];
convW=ArrayReshape[BinaryReadList[file,"Real32",outputLayers*inputLayers*w*h],{outputLayers,inputLayers,w,h}];
{convb,convW,convScales,convRM,convRV})


(* Naming Convention:
Layers follow the same structure as tiny YOLO, but index starts from 1, so for example CZ layer 5 corresponds to tiny YOLO layer 4.
And the convolution layer includes the convolution, batch normalization and leaky RELU
*)


{conv1b,conv1W,conv1Scales,conv1RM,conv1RV}=ReadConvolutionLayer[file,{16,3,3,3}];


CZConv1=ConvolutionLayer[16,{3,3},"Biases"->Table[0,{16}],"Weights"->conv1W,"PaddingSize"->1];


CZBN1=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{16,416,416},"Gamma"->conv1Scales,"Beta"->conv1b,"MovingMean"->conv1RM,"MovingVariance"->conv1RV];


{conv3b,conv3W,conv3Scales,conv3RM,conv3RV}=ReadConvolutionLayer[file,{32,16,3,3}];


CZConv3=ConvolutionLayer[32,{3,3},"Biases"->Table[0,{32}],"Weights"->conv3W,"PaddingSize"->1];


CZBN3=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{32,208,208},"Gamma"->conv3Scales,"Beta"->conv3b,"MovingMean"->conv3RM,"MovingVariance"->conv3RV];


{conv5b,conv5W,conv5Scales,conv5RM,conv5RV}=ReadConvolutionLayer[file,{64,32,3,3}];


CZConv5=ConvolutionLayer[64,{3,3},"Biases"->Table[0,{64}],"Weights"->conv5W,"PaddingSize"->1];


CZBN5=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{64,104,104},"Gamma"->conv5Scales,"Beta"->conv5b,"MovingMean"->conv5RM,"MovingVariance"->conv5RV];


{conv7b,conv7W,conv7Scales,conv7RM,conv7RV}=ReadConvolutionLayer[file,{128,64,3,3}];


CZConv7=ConvolutionLayer[128,{3,3},"Biases"->Table[0,{128}],"Weights"->conv7W,"PaddingSize"->1];


CZBN7=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{128,52,52},"Gamma"->conv7Scales,"Beta"->conv7b,"MovingMean"->conv7RM,"MovingVariance"->conv7RV];


{conv9b,conv9W,conv9Scales,conv9RM,conv9RV}=ReadConvolutionLayer[file,{256,128,3,3}];


CZConv9=ConvolutionLayer[256,{3,3},"Biases"->Table[0,{256}],"Weights"->conv9W,"PaddingSize"->1];


CZBN9=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{256,26,26},"Gamma"->conv9Scales,"Beta"->conv9b,"MovingMean"->conv9RM,"MovingVariance"->conv9RV];


{conv11b,conv11W,conv11Scales,conv11RM,conv11RV}=ReadConvolutionLayer[file,{512,256,3,3}];


CZConv11=ConvolutionLayer[512,{3,3},"Biases"->Table[0,{512}],"Weights"->conv11W,"PaddingSize"->1];


CZBN11=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{512,13,13},"Gamma"->conv11Scales,"Beta"->conv11b,"MovingMean"->conv11RM,"MovingVariance"->conv11RV];


{conv13b,conv13W,conv13Scales,conv13RM,conv13RV}=ReadConvolutionLayer[file,{1024,512,3,3}];


CZConv13=ConvolutionLayer[1024,{3,3},"Biases"->Table[0,{1024}],"Weights"->conv13W,"PaddingSize"->1];


CZBN13=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{1024,13,13},"Gamma"->conv13Scales,"Beta"->conv13b,"MovingMean"->conv13RM,"MovingVariance"->conv13RV];


{conv14b,conv14W,conv14Scales,conv14RM,conv14RV}=ReadConvolutionLayer[file,{1024,1024,3,3}];


CZConv14=ConvolutionLayer[1024,{3,3},"Biases"->Table[0,{1024}],"Weights"->conv14W,"PaddingSize"->1];


CZBN14=BatchNormalizationLayer["Epsilon"->.00001,"Input"->{1024,13,13},"Gamma"->conv14Scales,"Beta"->conv14b,"MovingMean"->conv14RM,"MovingVariance"->conv14RV];


conv15b=BinaryReadList[file,"Real32",125];
conv15W=ArrayReshape[BinaryReadList[file,"Real32",125*1024*1*1],{125,1024,1,1}];


CZConv15=ConvolutionLayer[125,{1,1},"Biases"->conv15b,"Weights"->conv15W];


biases={{1.08,1.19},{3.42,4.41},{6.63,11.38},{9.42,5.11},{16.62,10.52}};


CZGetBoundingBox[cubePos_]:=
(
   centX=
      (1/13)*(cubePos[[3]]-1+LogisticSigmoid[conv15[[1,1+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]);
   centY=
      (1/13)*(cubePos[[2]]-1+LogisticSigmoid[conv15[[1,2+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]);
   w=(1/13)*Exp[conv15[[1,3+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]*biases[[cubePos[[1]],1]];
   h=
      (1/13)*(Exp[conv15[[1,4+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]*biases[[cubePos[[1]],2]]);
   Rectangle[416*{centX-w/2,1-(centY+h/2)},416*{centX+w/2,1-(centY-h/2)}]
)


CorrectBox[box_,image_]:=
   If[ImageAspectRatio[image]<1,
         Rectangle[
            {box[[1,1]]*ImageDimensions[image][[1]]/416,box[[1,2]]*ImageDimensions[image][[1]]/416-(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])/2},
            {box[[2,1]]*ImageDimensions[image][[1]]/416,box[[2,2]]*ImageDimensions[image][[1]]/416-(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])/2}],
         Rectangle[
            {box[[1,1]]*ImageDimensions[image][[2]]/416-(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])/2,box[[1,2]]*ImageDimensions[image][[2]]/416},
            {box[[2,1]]*ImageDimensions[image][[2]]/416-(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])/2,box[[2,2]]*ImageDimensions[image][[2]]/416}]
   ]


object={"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};


CZMap[cubePos_]:={object[[cubePos[[4]]]],CZGetBoundingBox[cubePos]}


CZDisplayObject[object_]:={Rectangle@@object[[2]],Text[Style[object[[1]],White,24],{20,20}+object[[2,1]],Background->Black]}


CZNonMaxSuppression[objectsInClass_]:=
   Map[{objectsInClass[[1,1]],Rectangle[#[[1]],#[[2]]]}&,CZDeleteOverlappingWindows[Map[{#[[2]],#[[3,1]],#[[3,2]]}&,objectsInClass]]]


CZDetectObjects[img_]:=
   Flatten[Map[CZNonMaxSuppression,GatherBy[CZRawDetectObjects[img],#[[1]]&]],1]


CZRawDetectObjects[image_]:=(
   inp={ImageData[CZImagePadToSquare[CZMaxSideImage[image,416]],Interleaving->False]};
   conv1=CZConv1@inp[[1;;1,1;;3]];
   bn1=CZBN1@conv1;
   lr1=UnitStep[bn1]*bn1 + (1-UnitStep[bn1])*bn1*0.1;
   maxpool2=PoolingLayer[{2,2},"Stride"->2]@lr1;
   conv3=CZConv3@maxpool2;
   bn3=CZBN3@conv3;
   lr3=UnitStep[bn3]*bn3 + (1-UnitStep[bn3])*bn3*0.1;
   maxpool4=PoolingLayer[{2,2},"Stride"->2]@lr3;
   conv5=CZConv5@maxpool4;
   bn5=CZBN5@conv5;
   lr5=UnitStep[bn5]*bn5 + (1-UnitStep[bn5])*bn5*0.1;
   maxpool6=PoolingLayer[{2,2},"Stride"->2]@lr5;
   conv7=CZConv7@maxpool6;
   bn7=CZBN7@conv7;
   lr7=UnitStep[bn7]*bn7 + (1-UnitStep[bn7])*bn7*0.1;
   maxpool8=PoolingLayer[{2,2},"Stride"->2]@lr7;
   conv9=CZConv9@maxpool8;
   bn9=CZBN9@conv9;
   lr9=UnitStep[bn9]*bn9 + (1-UnitStep[bn9])*bn9*0.1;
   maxpool10=PoolingLayer[{2,2},"Stride"->2]@lr9;
   conv11=CZConv11@maxpool10;
   bn11=CZBN11@conv11;
   lr11=UnitStep[bn11]*bn11 + (1-UnitStep[bn11])*bn11*0.1;
   rs11={ArrayPad[lr11[[1]],{{0},{0,1},{0,1}},-100.]};
   maxpool12=(PoolingLayer[{2,2},"Stride"->1]@rs11);
   conv13=CZConv13@maxpool12;
   bn13=CZBN13@conv13;
   lr13=UnitStep[bn13]*bn13 + (1-UnitStep[bn13])*bn13*0.1;
   conv14=CZConv14@lr13;
   bn14=CZBN14@conv14;
   lr14=UnitStep[bn14]*bn14 + (1-UnitStep[bn14])*bn14*0.1;
   conv15=CZConv15@lr14;
   cube=Table[LogisticSigmoid[conv15[[1,5+(n*25),r,c]]]*
      SoftmaxLayer[][conv15[[1,6+(n*25);;6+(n*25)+20-1,r,c]]],
      {n,0,4},{r,1,13},{c,1,13}];
   extract=Position[cube,x_/;x>.24];
   Map[{object[[#[[4]]]],cube[[#[[1]],#[[2]],#[[3]],#[[4]]]],CorrectBox[CZMap[#][[2]],image]}&,extract]
)


(*
   Some sample test code that was used for cross checking results from the DarkNet codebase
   imgDat=BinaryReadList["/Users/julian/Downloads/darknet-master/img.dat","Real32",416*416*3];
   img=ArrayReshape[imgDat,{3,416,416}];img//Dimensions
   layerDat=BinaryReadList["/Users/julian/Downloads/darknet-master/layer.dat","Real32",13*13*125];
   layer=ArrayReshape[layerDat,{125,13,13}];layer//Dimensions
   diff=Abs[layer-conv15[[1]]];
*)
