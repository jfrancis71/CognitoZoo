(* ::Package:: *)

(* Implements SSD VGG 512 Pascal VOC on Mathematica version 11.

   SSD VGG 512 Pascal VOC is a computer vision object detection and localisation model designed to detect
   20 object categories (e.g. people, horses, dogs etc)
   Performance: mAP 82.2% on PascalVOC2012Test
   
   Usage: HighlightImage[img, CZDisplayObject /@ CZDetectObjects[img]]
   or:    CZHighlightObjects[ img ]

   Timings are around: (for the two cars on Clapham Common image)
   3200 secs for MacBook Air
*)

(*
   Credit:
   
   Wei Liu's CAFFE model was the reference model for this Mathematia implementation:
      https://github.com/weiliu89/caffe/tree/ssd (Downloaded 21/09/2018)
   and uses weights from VGG_VOC0712Plus_SSD_512x512_ft_iter_160000.caffemodel (Downloaded 18/10/2018)

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
   CZObjectsDeconformer[ img, {512, 512}, "Stretch" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (SSDNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[{512,512},"Stretch"]@img;


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


(* Private Implementation Code *)


(* The weights in this file have been converted from: https://github.com/weiliu89/caffe/tree/ssd
   See model reference: Pascal VOC 07++12+COCO SSD512
*)
SSDNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDVGG512VOC07Plus12FT.wlnet"],"WLNet"];


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[CZPascalClasses,detections[[All,2;;2]]],
      Extract[netOutput["ClassProb"], detections ]
   }]
];
