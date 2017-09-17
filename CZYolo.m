(* ::Package:: *)

(* Implements tiny YOLO on Mathematica version 11.

   tiny YOLO is a computer vision object detection and localisation model designed to detect
   20 object categories (e.g. people, horses, dogs etc)
   
   Usage: HighlightImage[img, CZDisplayObject /@ CZDetectObjects[img]]

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
   
   Takes about 1.25 secs to run on an image (MacBook air, CPU). (Darknet has reported tiny YOLO running at
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
      https://drive.google.com/file/d/0Bzhe0pgVZtNUWk9TOUdOTVJxMlk/view?usp=sharing
*)


(* Copyright Julian Francis 2017. Please see license file for details. *)


<<CZUtils.m


CZImagePadToSquare[image_Image]:=
   If[ImageAspectRatio[image]<1,
   ImagePad[image,{{0,0},{(1/2)*(ImageDimensions[image][[1]]-ImageDimensions[image][[2]]),Ceiling[(1/2)*(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])]}},Padding->0.5],
   ImagePad[image,{{(1/2)*(ImageDimensions[image][[2]]-ImageDimensions[image][[1]]),Ceiling[(1/2)*(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])]},{0,0}},Padding->0.5]]


file=OpenRead["CZModels/tiny-yolo-voc.weights",BinaryFormat->True];


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
      (1/13)*(cubePos[[3]]-1+LogisticSigmoid[conv15[[1+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]);
   centY=
      (1/13)*(cubePos[[2]]-1+LogisticSigmoid[conv15[[2+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]);
   w=(1/13)*Exp[conv15[[3+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]*biases[[cubePos[[1]],1]];
   h=
      (1/13)*(Exp[conv15[[4+(cubePos[[1]]-1)*25,cubePos[[2]],cubePos[[3]]]]]*biases[[cubePos[[1]],2]]);
   Rectangle[416*{centX-w/2,1-(centY+h/2)},416*{centX+w/2,1-(centY-h/2)}]
)


CZTransformRectangleToImage[rect_Rectangle,image_]:=
   If[ImageAspectRatio[image]<1,
         Rectangle[
            {rect[[1,1]]*ImageDimensions[image][[1]]/416,rect[[1,2]]*ImageDimensions[image][[1]]/416-(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])/2},
            {rect[[2,1]]*ImageDimensions[image][[1]]/416,rect[[2,2]]*ImageDimensions[image][[1]]/416-(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])/2}],
         Rectangle[
            {rect[[1,1]]*ImageDimensions[image][[2]]/416-(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])/2,rect[[1,2]]*ImageDimensions[image][[2]]/416},
            {rect[[2,1]]*ImageDimensions[image][[2]]/416-(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])/2,rect[[2,2]]*ImageDimensions[image][[2]]/416}]
   ]


CZTransformRectangleToYolo[rect_Rectangle, image_] :=
   If[ImageAspectRatio[image]<1,
         Rectangle[
            {rect[[1,1]]*416/ImageDimensions[image][[1]],416/ImageDimensions[image][[1]]*(rect[[1,2]]+(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])/2)},
            {rect[[2,1]]*416/ImageDimensions[image][[1]],416/ImageDimensions[image][[1]]*(rect[[2,2]]+(ImageDimensions[image][[1]]-ImageDimensions[image][[2]])/2)}],
         Rectangle[
            {416/ImageDimensions[image][[2]]*(rect[[1,1]]+(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])/2),rect[[1,2]]*416/ImageDimensions[image][[2]]},
            {416/ImageDimensions[image][[2]]*(rect[[2,1]]+(ImageDimensions[image][[2]]-ImageDimensions[image][[1]])/2),rect[[2,2]]*416/ImageDimensions[image][[2]]}]
   ]


CZpascalClasses = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
   "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};


CZMap[cubePos_]:={CZpascalClasses[[cubePos[[4]]]],CZGetBoundingBox[cubePos]}


CZDisplayObject[object_]:={Rectangle@@object[[2]],Text[Style[object[[1]],White,24],{20,20}+object[[2,1]],Background->Black]}


CZNonMaxSuppression[objectsInClass_]:=
   Map[{objectsInClass[[1,1]],Rectangle[#[[1]],#[[2]]]}&,CZDeleteOverlappingWindows[Map[{#[[2]],#[[3,1]],#[[3,2]]}&,objectsInClass]]]


CZDetectObjects[img_]:=
   Flatten[Map[CZNonMaxSuppression,GatherBy[CZRawDetectObjects[img],#[[1]]&]],1]


leakyReLU = ElementwiseLayer[Ramp[#]+Ramp[-#]*-.1&];


CZYoloNet = NetChain[{
   CZConv1,CZBN1,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv3,CZBN3,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv5,CZBN5,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv7,CZBN7,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv9,CZBN9,leakyReLU,PoolingLayer[{2,2},"Stride"->2],
   CZConv11,CZBN11,leakyReLU,PaddingLayer[{{0,0},{0,1},{0,1}},-100],PoolingLayer[{2,2},"Stride"->1],
   CZConv13,CZBN13,leakyReLU,
   CZConv14,CZBN14,leakyReLU,
   CZConv15},
   "Input"->NetEncoder[{"Image",{416,416},ColorSpace->"RGB"}]];


CZRawDetectObjects[image_]:=(
   conv15=
     CZYoloNet[CZImagePadToSquare[image]];
   cube=LogisticSigmoid[conv15[[5;;105;;25]]]*SoftmaxLayer[][Transpose[Partition[conv15,25][[All,6;;25]],{1,4,2,3}]];
   extract=Position[cube,x_/;x>.24];
   Map[{CZpascalClasses[[#[[4]]]],cube[[#[[1]],#[[2]],#[[3]],#[[4]]]],CZTransformRectangleToImage[CZMap[#][[2]],image]}&,extract]
)


(*
   Below functions not necessary for Yolo detection, but useful for inspecting the output of the
   net and are differentiable.
*)


(*
Little net that can be attached to end of YoloNet to extract a particular slot
Usage: YoloSlotExtractNet[8,4,2] note slot runs from 1 to 5.
*)
YoloSlotExtractNet[row_,col_,slot_] :=
   NetChain[{TransposeLayer[1->3],TransposeLayer[],PartLayer[row],PartLayer[col],PartLayer[(slot-1)*25+1;;(slot*25)]}];
   
SlotToObjProbNet[object_] := 
   NetGraph[{PartLayer[5],ElementwiseLayer[LogisticSigmoid],PartLayer[5+1;;5+20],SoftmaxLayer[],PartLayer[object],ThreadingLayer[Times]},{1->2,3->4,4->5,5->6,2->6}];

(* Suggested use:
   NetChain[{CZYoloNet,YoloSlotExtractNet[8,4,2],SlotToObjProbNet[7]}][CZImagePadToSquare[CZMaxSideImage[img,416]]]
*)

SlotToSoftMax=NetChain[{PartLayer[6;;25],TransposeLayer[3->1],TransposeLayer[],SoftmaxLayer[]}];
(* Slightly unfortunate implementation, would be better if stack spported broadcasting *)
SlotToProb=NetGraph[{SlotToSoftMax,PartLayer[5],LogisticSigmoid,ReplicateLayer[20],TransposeLayer[3->1],TransposeLayer[],ThreadingLayer[Times]},{2->3,3->4,4->5,5->6,6->7,1->7}];
SlotsToProb=NetGraph[{
PartLayer[1;;25],SlotToProb,
PartLayer[26;;50],SlotToProb,
PartLayer[51;;75],SlotToProb,
PartLayer[76;;100],SlotToProb,
PartLayer[101;;125],SlotToProb,
ThreadingLayer[Plus],
ThreadingLayer[Plus],
ThreadingLayer[Plus],
ThreadingLayer[Plus]},{1->2,3->4,5->6,7->8,9->10,2->11,4->11,6->12,8->12,11->13,12->13,10->14,13->14}];


(*
   YoloGetMaxPatch takes a YOLO normalised image (ie 416x416) and a neural network output layer.
   It can then dot prod this with a weight vector and returns the image patch which maximises this dot prob
   receptiveFieldSize is 38 for conv7, 78 for conv9
*)
YoloMaxPatch[{image_,layer_},attrib_,receptiveFieldSize_]:=(
   map=Transpose[layer,{3,1,2}].attrib;
   p=Position[map,Max[map]][[1]];
   {y,x}=p*{416,416}/Length[layer[[1,1]]];
   {
      Max[map],
      ImageTake[ImagePad[image,receptiveFieldSize/2],{y,y+receptiveFieldSize},{x,x+receptiveFieldSize}]
   }
)


(*
   Some sample test code that was used for cross checking results from the DarkNet codebase
   imgDat=BinaryReadList["/Users/julian/Downloads/darknet-master/img.dat","Real32",416*416*3];
   img=ArrayReshape[imgDat,{3,416,416}];img//Dimensions
   layerDat=BinaryReadList["/Users/julian/Downloads/darknet-master/layer.dat","Real32",13*13*125];
   layer=ArrayReshape[layerDat,{125,13,13}];layer//Dimensions
   diff=Abs[layer-conv15[[1]]];
*)
