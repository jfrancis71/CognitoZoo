(* ::Package:: *)

(* Implements AlexNet

   You will need to download:
      class_names.json from https://drive.google.com/open?id=0Bzhe0pgVZtNUc2RVVVFNSVF0NW8
      alexnet.hdf from https://drive.google.com/open?id=0Bzhe0pgVZtNUNm1OdFFWWS1mb2c
   and install it on a Mathematica search path, eg your home directory.
   

   Credit:
      The following tensorflow code was used as a reference implementation:
      Michael Guerzhoy and Davi Frossard, 2016
      http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
      Model from  https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet
         License: This model is released for unrestricted use. (as at 12/03/2017)
      Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow
      
      Weights were loaded using above code, and then saved in HDF5 format.
*)


(* Public Interface Code *)


CZImageIdentify[image_] :=
   CZFixedSizeImageIdentify[ ImageCrop[ImageResize[ColorConvert[image,"RGB"],227],{227,227}] ]


(* Private Implementation Code *)


classNames=Import["CZModels/alexnet_names.json"];


(* Loading in CNN parameters *)
CZAlexNetFilename = "CZModels/alexnet.hdf";
conv1W=Import[CZAlexNetFilename,{"Datasets","/conv1W"}];
conv1b=Import[CZAlexNetFilename,{"Datasets","/conv1b"}];
conv2W=Import[CZAlexNetFilename,{"Datasets","/conv2W"}];
conv2b=Import[CZAlexNetFilename,{"Datasets","/conv2b"}];
conv3W=Import[CZAlexNetFilename,{"Datasets","/conv3W"}];
conv3b=Import[CZAlexNetFilename,{"Datasets","/conv3b"}];
conv4W=Import[CZAlexNetFilename,{"Datasets","/conv4W"}];
conv4b=Import[CZAlexNetFilename,{"Datasets","/conv4b"}];
conv5W=Import[CZAlexNetFilename,{"Datasets","/conv5W"}];
conv5b=Import[CZAlexNetFilename,{"Datasets","/conv5b"}];
fc6W=Import[CZAlexNetFilename,{"Datasets","/fc6W"}];
fc6b=Import[CZAlexNetFilename,{"Datasets","/fc6b"}];
fc7W=Import[CZAlexNetFilename,{"Datasets","/fc7W"}];
fc7b=Import[CZAlexNetFilename,{"Datasets","/fc7b"}];
fc8W=Import[CZAlexNetFilename,{"Datasets","/fc8W"}];
fc8b=Import[CZAlexNetFilename,{"Datasets","/fc8b"}];


CNConv1=ConvolutionLayer[96,{11,11},"Biases"->conv1b,"Weights"->Transpose[conv1W,{3,4,2,1}],"PaddingSize"->5];
CNConv2P1=ConvolutionLayer[128,{5,5},"Biases"->Table[0,{128}],"Weights"->Transpose[conv2W[[All,All,All,1;;128]],{3,4,2,1}],"PaddingSize"->2];
CNConv2P2=ConvolutionLayer[128,{5,5},"Biases"->Table[0,{128}],"Weights"->Transpose[conv2W[[All,All,All,129;;256]],{3,4,2,1}],"PaddingSize"->2];
CNConv3=ConvolutionLayer[384,{3,3},"Biases"->conv3b,"Weights"->Transpose[conv3W,{3,4,2,1}],"PaddingSize"->1];
CNConv4P1=ConvolutionLayer[192,{3,3},"Biases"->Table[0,{192}],"Weights"->Transpose[conv4W[[All,All,All,;;192]],{3,4,2,1}],"PaddingSize"->1];
CNConv4P2=ConvolutionLayer[192,{3,3},"Biases"->Table[0,{192}],"Weights"->Transpose[conv4W[[All,All,All,193;;]],{3,4,2,1}],"PaddingSize"->1];
CNConv5P1=ConvolutionLayer[128,{3,3},"Biases"->Table[0,{128}],"Weights"->Transpose[conv5W[[All,All,All,;;128]],{3,4,2,1}],"PaddingSize"->1];
CNConv5P2=ConvolutionLayer[128,{3,3},"Biases"->Table[0,{128}],"Weights"->Transpose[conv5W[[All,All,All,129;;]],{3,4,2,1}],"PaddingSize"->1];
FC6=DotPlusLayer[ "Weights"->Transpose[fc6W],"Biases"->fc6b];
FC7=DotPlusLayer[ "Weights"->Transpose[fc7W],"Biases"->fc7b];
FC8=DotPlusLayer[ "Weights"->Transpose[fc8W],"Biases"->fc8b];


LRN[layer_] := Module[{sq=Table[Sum[layer[[d1]]^2,{d1,Max[1,d-2],Min[Length[layer],d+2]}],{d,1,Length[layer]}]},layer/(1 +(2 10^-5)*sq)^0.75]


CZFixedSizeImageIdentify[image_] := (
   in4=256*(Transpose[ImageData[image][[All,All,1;;3]],{2,3,1}]-Mean[ImageData[image][[All,All,1;;3]]//Flatten]);
   (* Note that CAFFE striding started at index 2, not 1*)
   conv1=(NetChain[{CNConv1,Ramp}]@in4)[[All,2;;-1;;4,2;;-1;;4]];
   lrn1=LRN[conv1];
   maxpool1=PoolingLayer[{3,3},"Stride"->2]@lrn1;
   conv2p1=CNConv2P1@(maxpool1[[1;;48]]);
   conv2p2=CNConv2P2@(maxpool1[[49;;96]]);
   conv2=Ramp@MapThread[((z=#1)+#2)&,{Join[conv2p1,conv2p2],conv2b}];
   lrn2=LRN[conv2];
   maxpool2=(PoolingLayer[{3,3},"Stride"->2]@lrn2);
   conv3=(NetChain[{CNConv3,Ramp}]@maxpool2);
   conv4p1=CNConv4P1@conv3[[;;192]];
   conv4p2=CNConv4P2@conv3[[193;;]];
   conv4=Ramp@MapThread[((z=#1)+#2)&,{Join[conv4p1,conv4p2],conv4b}];
   conv5p1=CNConv5P1@(conv4[[;;192]]);
   conv5p2=CNConv5P2@(conv4[[193;;]]);   
   conv5=Ramp@MapThread[((z=#1)+#2)&,{Join[conv5p1,conv5p2],conv5b}];
   maxpool5=PoolingLayer[{3,3},"Stride"->2]@conv5;
   fc6=NetChain[{FC6,Ramp}]@Flatten[Transpose[maxpool5,{3,1,2}]];
   fc7=NetChain[{FC7,Ramp}]@fc6;
   fc8=FC8@fc7;
   final=SoftmaxLayer[]@fc8;
   classNames[[Position[final,Max[final]][[1,1]]]]
)
