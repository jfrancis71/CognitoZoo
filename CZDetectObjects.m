(* ::Package:: *)

CZDetectionNets = { "MobileNet", "SSDVGG512COCO", "RetinaNet", "SSDVGG300Pascal", "SSDVGG512Pascal" };


Options[ CZDetectObjects ] = {
   TargetDevice->"CPU",
   AcceptanceThreshold->.6, (* Note some default thresholds can vary in reference tests, but a good default choice *)
   MaxOverlapFraction->.45,
   NMSMethod->CZNonMaxSuppression
};
CZDetectObjects[ image_,  opts:OptionsPattern[ { CZDetectObjects, Method->"MobileNet" } ] ] := Switch[ OptionValue[ Method ],
   "SSDVGG512COCO", CZDetectObjectsSSDVGG512COCO[ image, FilterRules[ {opts}, Options[  CZDetectObjects ] ] ],
   "RetinaNet", CZDetectObjectsRetinaNet[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "MobileNet", CZDetectObjectsMobileNet[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "SSDVGG300Pascal", CZDetectObjectsSSDVGG300Pascal[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "SSDVGG512Pascal", CZDetectObjectsSSDVGG512Pascal[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ]
];


CZHighlightObjects[ img_Image, opts:OptionsPattern[ CZDetectObjects ] ] := HighlightImage[
   img,
   CZDisplayObjects@CZDetectObjects[ img, opts ]];


CZDetectObjectsSSDVGG512COCO[ image_, opts:OptionsPattern[ CZDetectObjects]  ] := CZDetectObjectsGeneric[ image, SSDVGG512COCONet, {512,512}, "Stretch", CZCOCOClasses, opts ];


CZDetectObjectsRetinaNet[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] := CZDetectObjectsGeneric[ image, RetinaNetR101FPNLR2Net, {1152,896}, "Fit", CZCOCOClasses, opts ];


CZDetectObjectsMobileNet[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] := CZDetectObjectsGeneric[ image, SSDMobileNetv2, {300,300}, "Stretch", CZCOCOClasses, opts ];


CZDetectObjectsSSDVGG300Pascal[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] := CZDetectObjectsGeneric[ image, SSDVGG300PascalNet, {300,300}, "Stretch", CZPascalClasses, opts ];


CZDetectObjectsSSDVGG512Pascal[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] := CZDetectObjectsGeneric[ image, SSDVGG512PascalNet, {512,512}, "Stretch", CZPascalClasses, opts ];


<<CognitoZoo/CZUtils.m


CZDetectObjectsGeneric[ img_Image, net_, netDims_, fitting_, labels_, opts:OptionsPattern[ CZDetectObjects ] ] :=
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ img, netDims, fitting ]@CZOutputDecoder[ labels, OptionValue[ AcceptanceThreshold ] ]@
   (net[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[ netDims, fitting]@img;


CZOutputDecoder[ labels_, threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[labels,detections[[All,2;;2]]],
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
SSDMobileNetv2 = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDMobileNetv2.wlnet"],"WLNet"];


(* The weights in the following wlnet file have been converted from: https://github.com/weiliu89/caffe/tree/ssd
   See model reference: Pascal VOC 07+12 SSD300
   Copyright and license details: https://github.com/weiliu89/caffe/blob/ssd/LICENSE
*)
SSDVGG300PascalNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDVGG300PascalVOCReference20180920.wlnet"],"WLNet"];


(* The weights in this file have been converted from: https://github.com/weiliu89/caffe/tree/ssd
   See model reference: Pascal VOC 07++12+COCO SSD512
   Copyright and license details: https://github.com/weiliu89/caffe/blob/ssd/LICENSE
*)
SSDVGG512PascalNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/SSDVGG512VOC07Plus12FT.wlnet"],"WLNet"];
