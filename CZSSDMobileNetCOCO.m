(* ::Package:: *)

(* Implements SSD MobileNet 300 COCO on Mathematica version 12.

   SSD MobileNet 300 COCO is a computer vision object detection and localisation model designed to detect
   80 object categories (e.g. people, horses, dogs etc)
   Performance: mAP 22% on COCO test-dev 2015 (using COCO performance metric)
   
   Usage: HighlightImage[img, CZDisplayObject /@ CZDetectObjects[img]]
   or:    CZHighlightObjects[ img ]

   Timings are around: (for the two cars on Clapham Common image)
   0.3 secs on Toshiba CPU
*)

(*
   Credit:
   
   Tensorflow Detection Model Zoo was the reference model for this Mathematica implementation:
      https://https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md (Downloaded 06/11/2018)
   and uses weights from ssd_mobilenet_v2_coco_2018_03_29 (Downloaded 06/11/2018)

   SSD MobileNet 300 is based on the following paper:
   https://https://arxiv.org/pdf/1801.04381.pdf
   Title: MobileNetV2: Inverted Residuals and Linear Bottlenecks
   Authors: Mark Sandler Andrew Howard Menglong Zhu Andrey Zhmoginov Liang-Chieh Chen
   Year: 2018
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


CZCOCOClasses = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
"zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
"fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
"cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"};


(* Private Implementation Code *)


SSDNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDMobileNetv2.wlnet"],"WLNet"];


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[Transpose@Transpose@netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[CZCOCOClasses,detections[[All,2;;2]]],
      Extract[netOutput["ClassProb"], detections ]
   }]
];
