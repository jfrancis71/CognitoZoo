(* ::Package:: *)

(* Implements SSD VGG 512 COCO on Mathematica version 11.

   SSD VGG 512 COCO is a computer vision object detection and localisation model designed to detect
   80 object categories (e.g. people, horses, dogs etc)
   Performance: mAP 28.8% on COCO test-dev 2015 (using COCO performance metric)
   
   Usage: HighlightImage[img, CZDisplayObject /@ CZDetectObjects[img]]
   or:    CZHighlightObjects[ img ]

   Timings are around: (for the two cars on Clapham Common image)
   4.3 secs for MacBook Air
*)

(*
   Credit:
   
   Wei Liu's CAFFE model was the reference model for this Mathematia implementation:
      https://github.com/weiliu89/caffe/tree/ssd (Downloaded 21/09/2018)
   and uses weights from VGG_coco_SSD_512x512_iter_360000.caffemodel (Downloaded 14/10/2018)

   SSD VGG 512 is based on the following paper:
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


CZCOCOClasses = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
"zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
"fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
"cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"};


(* Private Implementation Code *)


(* The weights in this file have been converted from: https://github.com/weiliu89/caffe/tree/ssd
   See model reference: COCO models trainval35K SSD512*
*)
SSDNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDVGG512COCOReference.wlnet"],"WLNet"];


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[CZCOCOClasses,detections[[All,2;;2]]],
      Extract[netOutput["ClassProb"], detections ]
   }]
];
