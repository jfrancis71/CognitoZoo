(* ::Package:: *)

Options[ CZDetectObjectsDefaults ] = {
   TargetDevice->"CPU",
   NMSMethod->CZNonMaxSuppression
};
Options[ CZDetectObjects ] = Join[
   Options[ CZDetectObjectsDefaults ],{
   Threshold->Automatic,
   NMSIntersectionOverUnionThreshold->Automatic,
   Method->"SSDVGG512COCO"
}];
CZDetectObjects[ image_,  opts:OptionsPattern[] ] := Switch[ OptionValue[ Method ],
   "SSDVGG512COCO", CZDetectObjectsSSDVGG512COCO[ image, FilterRules[ {opts}, Options[  CZDetectObjectsSSDVGG512COCO ] ] ],
   "RetinaNet", CZDetectObjectsRetinaNet[ image, FilterRules[ {opts}, Options[ CZDetectObjectsRetinaNet ] ] ],
   "MobileNet", CZDetectObjectsMobileNet[ image, FilterRules[ {opts}, Options[ CZDetectObjectsMobileNet ] ] ]
];


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := HighlightImage[
   img,
   CZDisplayObject /@ CZDetectObjects[ img, opts ]];


Options[ CZDetectObjectsSSDVGG512COCO ] = Join[
   Options[ CZDetectObjectsDefaults ],{
   Threshold->0.6,
   NMSIntersectionOverUnionThreshold->.45
}];
CZDetectObjectsSSDVGG512COCO[ image_, opts:OptionsPattern[]  ] := CZDetectObjectsGeneric[ image, SSDVGG512COCONet, {512,512}, "Stretch", opts ];


Options[ CZDetectObjectsRetinaNet ] = Join[
   Options[ CZDetectObjectsDefaults ],{
   Threshold->0.6,
   NMSIntersectionOverUnionThreshold->.45
}];
CZDetectObjectsRetinaNet[ image_, opts:OptionsPattern[]  ] := CZDetectObjectsGeneric[ image, RetinaNetR101FPNLR2Net, {1152,896}, "Fit", opts ];


Options[ CZDetectObjectsMobileNet ] = Join[
   Options[ CZDetectObjectsDefaults ],{
   Threshold->0.6,
   NMSIntersectionOverUnionThreshold->.45
}];
CZDetectObjectsMobileNet[ image_, opts:OptionsPattern[]  ] := CZDetectObjectsGeneric[ image, SSDMobileNetv2, {300,300}, "Stretch", opts ];


<<CognitoZoo/CZUtils.m


Options[ CZDetectObjectsGeneric ] = Join[
   Options[ CZDetectObjectsDefaults ],{
   Threshold->0.6,
   NMSIntersectionOverUnionThreshold->.45
}];
CZDetectObjectsGeneric[ img_Image, net_, netDims_, fitting_, opts:OptionsPattern[] ] :=
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ img, netDims, fitting ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (net[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[ netDims, fitting]@img;


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[CZCOCOClasses,detections[[All,2;;2]]],
      Extract[netOutput["ClassProb"], detections ]
   }]
];


CZCOCOClasses = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
"zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
"fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
"cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"};


(* The weights in this file have been converted from: https://github.com/weiliu89/caffe/tree/ssd
   See model reference: COCO models trainval35K SSD512*
   Copyright and license details: https://github.com/weiliu89/caffe/blob/ssd/LICENSE
*)
SSDVGG512COCONet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDVGG512COCOReference.wlnet"],"WLNet"];


(*

Weights in below file are converted from:
Facebook Detectron model: R-101-FPN LRN 2
https: https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md

@misc{Detectron2018,
  author =       {Ross Girshick and Ilija Radosavovic and Georgia Gkioxari and
                  Piotr Doll\'{a}r and Kaiming He},
  title =        {Detectron},
  howpublished = {\url{https://github.com/facebookresearch/detectron}},
  year =         {2018}
*)
RetinaNetR101FPNLR2Net = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/RetinaNetR101FPNLR2.wlnet"],"WLNet"];


(*
Weights in below file converted from:
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

*)
SSDMobileNetv2 = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDMobileNetv2.wlnet"],"WLNet"];
