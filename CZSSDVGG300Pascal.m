(* ::Package:: *)

(* Implements SSD VGG 300 Pascal VOC on Mathematica version 11.

   SSD VGG 300 Pascal VOC is a computer vision object detection and localisation model designed to detect
   20 object categories (e.g. people, horses, dogs etc)
   Performance: mAP 77.2% on PascalVOC2007Test
   
   Usage: HighlightImage[img, CZDisplayObject /@ CZDetectObjects[img]]
   or:    CZHighlightObjects[ img ]

   Timings are around: (for the two cars on Clapham Common image)
   1.3 secs for MacBook Air
   1.1 secs Desktop CPU
   .34 secs Desktop GPU
*)

(*
   Credit:
   
   Wei Liu's CAFFE model was the reference model for this Mathematia implementation:
      https://github.com/weiliu89/caffe/tree/ssd (Downloaded 21/09/2018)
   and uses weights from VGG_VOC0712_SSD_300x300_iter_120000.caffemodel (Downloaded 20/09/2018)

   SSD VGG 300 is based on the following paper:
   https://arxiv.org/abs/1512.02325
   Title: SSD: Single Shot MultiBox Detector
   Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
   Cheng-Yang Fu, Alexander C. Berg
   Year: 2016
*)


(* Copyright Julian Francis 2018. Please see license file for details. *)


(* Public Interface Code *)


<<CZUtils.m


Options[ CZDetectObjects ] = Join[{
   TargetDevice->"CPU",
   Threshold->.6,(* This is the Wei Liu default setting for this implementation *)
   NMSIntersectionOverUnionThreshold->.45 (* This is the Wei Liu default setting for this implementation *)
}, Options[ CZNonMaxSuppressionPerClass ] ];
CZDetectObjects[ img_Image, opts:OptionsPattern[] ] :=
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ img, {300, 300}, "Stretch" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (SSDNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[{300,300},"Stretch"]@img;


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


(* Private Implementation Code *)


(* The weights in the following wlnet file have been converted from: https://github.com/weiliu89/caffe/tree/ssd
   See model reference: Pascal VOC 07+12 SSD300
   Copyright and license details: https://github.com/weiliu89/caffe/blob/ssd/LICENSE
*)
SSDNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDVGG300PascalVOCReference20180920.wlnet"],"WLNet"];


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[CZPascalClasses,detections[[All,2;;2]]],
      Extract[netOutput["ClassProb"], detections ]
   }]
];
