(* ::Package:: *)

(* Implements tiny YOLO on Mathematica version 11.

   tiny YOLO is a computer vision object detection and localisation model designed to detect
   20 object categories (e.g. people, horses, dogs etc)
   
   Usage: HighlightImage[img, CZDisplayObjects@CZDetectObjects[img]]
   or:    CZHighlightObjects[ img ]
      
   Takes about .28 secs to run on an image (MacBook air, CPU). (Darknet has reported tiny YOLO running at
   over 100 frames/sec).
   Desktop CPU: .34 secs
   Desktop GPU: .11 secs

*)

(*
   Credit:
   
   The code is based on the tiny YOLO model from Darknet, Joseph Redmon:
      https://pjreddie.com/darknet/yolo/
      
      Citation:
      @misc{darknet13,
      author =   {Joseph Redmon},
      title =    {Darknet: Open Source Neural Networks in C},
      howpublished = {\url{http://pjreddie.com/darknet/}},
      year = {2013--2016}
      }      
   
*)


(*

   The net file uses weights from the pretrained weight file from Darknet (http://pjreddie.com/darknet/)

   You will need to download the tiny YOLO WLNet file and install it on a Mathematica search path, eg your home directory.
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


   Original drknet weight file available here (as of 27 Nov 2016):
      https://pjreddie.com/media/files/tiny-yolo-voc.weights
      
   In case it changes, I have an additional copy here:
      https://drive.google.com/file/d/0Bzhe0pgVZtNUWk9TOUdOTVJxMlk/view?usp=sharing
*)


(* Copyright Julian Francis 2017. Please see license file for details. *)


(* Public Interface Code *)


<<CZUtils.m


Options[ CZDetectObjects ] = Join[{
   TargetDevice->"CPU",
   AcceptanceThreshold->.24
}, Options[ CZNonMaxSuppressionPerClass ] ];
CZDetectObjects[ image_, opts:OptionsPattern[] ] :=
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ image, {416, 416}, "Fit" ]@CZOutputDecoder[ OptionValue[ AcceptanceThreshold ] ]@
   (YoloNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@CZImageConformer[{416,416},"Fit"]@image;


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_,  opts:OptionsPattern[] ] := HighlightImage[img, CZDisplayObjects@CZDetectObjects[img, opts]]


(* Private Implementation Code *)


biases={{1.08,1.19},{3.42,4.41},{6.63,11.38},{9.42,5.11},{16.62,10.52}};


CZGetBoundingBox[ cubePos_, conv15_ ]:=
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

(* The weights in the following wlnet file have been converted from: https://pjreddie.com/media/files/tiny-yolo-voc.weights
   Copyright and license details: https://github.com/pjreddie/darknet/blob/master/LICENSE.mit
*)
YoloNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/TinyYolov2.wlnet"],"WLNet"];


CZOutputDecoder[ threshold_:.24 ][ netOutput_ ] :=(
   slots = LogisticSigmoid[netOutput[[5;;105;;25]]]*SoftmaxLayer[][Transpose[Partition[netOutput,25][[All,6;;25]],{1,4,2,3}]];
   slotPositions = Position[slots, x_/;x>threshold];
   Map[{CZGetBoundingBox[#,netOutput],CZPascalClasses[[#[[4]]]],slots[[#[[1]],#[[2]],#[[3]],#[[4]]]]}&,slotPositions]);


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
