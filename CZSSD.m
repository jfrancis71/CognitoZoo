(* ::Package:: *)

(* Implements SSD VGG 300 on Mathematica version 11.

   SSD VGG 300 is a computer vision object detection and localisation model designed to detect
   20 object categories (e.g. people, horses, dogs etc)
   
   Usage: HighlightImage[img, CZDisplayObject /@ CZDetectObjects[img]]
   or:    CZHighlightObjects[ img ]

   Timings are around: (for the two cars on Clapham Common image)
   1.3 secs for MacBook Air
   1.1 secs Desktop CPU
   .34 secs Desktop GPU
*)

(*
   Credit:
   This implementation is based on Changan Wang's Tensorflow code:
      https://github.com/HiKapok/SSD.TensorFlow
   
   The neural net file: SSDVGG300HiKapok.wlnet
   has been converted from TensorFlow checkpoint to Mathematica WLNET format
   The file is used with permission of the Apache 2.0 license:
   https://github.com/HiKapok/SSD.TensorFlow/LICENSE

   SSD VGG 300 is based on the following paper:
   https://arxiv.org/abs/1512.02325
   Title: SSD: Single Shot MultiBox Detector
   Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
   Cheng-Yang Fu, Alexander C. Berg
*)


(* Copyright Julian Francis 2018. Please see license file for details. *)


(* Public Interface Code *)


<<CZutils.m


Options[ CZDetectObjects ] = Join[{
   TargetDevice->"CPU",
   Threshold->.5
}, Options[ CZNonMaxSuppressionPerClass ] ];
CZDetectObjects[ image_Image, opts:OptionsPattern[] ] :=
   (
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ image, {300, 300}, "Stretch" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (SSDNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[{300,300},"Stretch"]@image
   )[[All,{1,3}]];


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


(* Private Implementation Code *)


SSDNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSD300VGGHikapokNet20180618.wlnet"],"WLNet"];


CZDecodeOutput[ locs_, probs_, threshold_:.5 ]:=Module[{
   detections = Position[probs,x_/;x>threshold]},

   MapThread[{#1,#2,{},Rectangle[300*{#3-#5/2,1-(#4+#6/2)},300*{#3+#5/2,1-(#4-#6/2)}]}&,{
      CZPascalClasses[[detections[[All,4]]]],
      Extract[probs,detections],
      Extract[locs[[1]],detections[[All,1;;3]]], (*cx*)
      Extract[locs[[2]],detections[[All,1;;3]]], (*cy*)
      Extract[locs[[3]],detections[[All,1;;3]]], (*width*)
      Extract[locs[[4]],detections[[All,1;;3]]]  (*height*)
}]
]


CZOutputDecoder[ threshold_:.5 ] := Function[ { netOutput },
   Join[
      CZDecodeOutput[netOutput["Locs1"],netOutput["ObjMap1"][[All,All,All,2;;21]], threshold],
      CZDecodeOutput[netOutput["Locs2"],netOutput["ObjMap2"][[All,All,All,2;;21]], threshold],
      CZDecodeOutput[netOutput["Locs3"],netOutput["ObjMap3"][[All,All,All,2;;21]], threshold],
      CZDecodeOutput[netOutput["Locs4"],netOutput["ObjMap4"][[All,All,All,2;;21]], threshold],
      CZDecodeOutput[netOutput["Locs5"],netOutput["ObjMap5"][[All,All,All,2;;21]], threshold],
      CZDecodeOutput[netOutput["Locs6"],netOutput["ObjMap6"][[All,All,All,2;;21]], threshold]
] ];
