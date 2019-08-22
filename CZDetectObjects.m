(* ::Package:: *)

CZDetectionNets = { "MobileNet", "SSDVGG512COCO", "RetinaNet", "SSDVGG300Pascal", "SSDVGG512Pascal", "TinyYolo", "Yolo9000" };


Options[ CZDetectObjects ] = {
   TargetDevice->"CPU",
   AcceptanceThreshold->"ModelSpecific", (* Note some default thresholds can vary in reference tests, but a good default choice *)
   MaxOverlapFraction->.45,
   NMSMethod->CZNonMaxSuppression
};
CZDetectObjects::method = "CZDetectObjects method `1` should be one of "<>StringRiffle[ CZDetectionNets, ", " ];
CZDetectObjects[ image_Image,  opts:OptionsPattern[ { CZDetectObjects, Method->"MobileNet" } ] ] := Switch[ OptionValue[ Method ],
   "SSDVGG512COCO", CZDetectObjectsSSDVGG512COCO[ image, FilterRules[ {opts}, Options[  CZDetectObjects ] ] ],
   "RetinaNet", CZDetectObjectsRetinaNet[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "MobileNet", CZDetectObjectsMobileNet[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "SSDVGG300Pascal", CZDetectObjectsSSDVGG300Pascal[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "SSDVGG512Pascal", CZDetectObjectsSSDVGG512Pascal[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "TinyYolo", CZDetectObjectsTinyYolo[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   "Yolo9000", CZDetectObjectsYolo9000[ image, FilterRules[ {opts}, Options[ CZDetectObjects ] ] ],
   _, Message[ CZDetectObjects::method, OptionValue[ Method ] ]
];


CZHighlightObjects[ img_Image, opts:OptionsPattern[ CZDetectObjects ] ] := HighlightImage[
   img,
   CZDisplayObjects[ CZDetectObjects[ img, opts ], CZCmap ] ];


CZDetectObjectsSSDVGG512COCO[ image_, opts:OptionsPattern[ CZDetectObjects]  ] :=
   CZDetectObjectsSingleStage[ image, SSDVGG512COCONet, CZLogisticOutputDecoder, OptionValue[ AcceptanceThreshold ] /. "ModelSpecific"->.6, {512,512}, "Stretch", 0, CZCOCOClasses, opts ];


CZDetectObjectsRetinaNet[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] :=
   CZDetectObjectsSingleStage[ image, RetinaNetR101FPNLR2Net, CZLogisticOutputDecoder, OptionValue[ AcceptanceThreshold ] /. "ModelSpecific"->.6, {1152,896}, "Fit", 0, CZCOCOClasses, opts ];


CZDetectObjectsMobileNet[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] :=
   CZDetectObjectsSingleStage[ image, SSDMobileNetv2, CZLogisticOutputDecoder, OptionValue[ AcceptanceThreshold ] /. "ModelSpecific"->.6, {300,300}, "Stretch", 0, CZCOCOClasses, opts ];


CZDetectObjectsSSDVGG300Pascal[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] :=
   CZDetectObjectsSingleStage[ image, SSDVGG300PascalNet, CZLogisticOutputDecoder, OptionValue[ AcceptanceThreshold ] /. "ModelSpecific"->.6, {300,300}, "Stretch", 0, CZPascalClasses, opts ];


CZDetectObjectsSSDVGG512Pascal[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] :=
   CZDetectObjectsSingleStage[ image, SSDVGG512PascalNet, CZLogisticOutputDecoder, OptionValue[ AcceptanceThreshold ] /. "ModelSpecific"->.6, {512,512}, "Stretch", 0, CZPascalClasses, opts ];


CZDetectObjectsTinyYolo[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] :=
   CZDetectObjectsSingleStage[ image, TinyYoloNet, CZTinyYoloOutputDecoder, OptionValue[ AcceptanceThreshold ] /. "ModelSpecific"->.24, {416,416}, "Fit", 0, CZPascalClasses, opts ];


CZDetectObjectsYolo9000[ image_, opts:OptionsPattern[ CZDetectObjects ]  ] :=
   CZDetectObjectsSingleStage[ image, yolo9000Net, CZYolo9000OutputDecoder, OptionValue[ AcceptanceThreshold ] /. "ModelSpecific"->.24, {544,544}, "Fit", 0.5, yolo9000Names, opts ];


<<CognitoZoo/CZUtils.m


CZDetectObjectsSingleStage[ img_Image, net_, outputDecoder_, threshold_, netDims_, fitting_, padding_, labels_, opts:OptionsPattern[ CZDetectObjects ] ] :=
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ img, netDims, fitting ]@outputDecoder[ labels, threshold ]@
   (net[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[ netDims, fitting, Padding->padding ]@img;


CZLogisticOutputDecoder[ labels_, threshold_:.5 ][ netOutput_ ] := Module[{
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

CZPascalClasses = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
   "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

SeedRandom[1];
Map[ (CZCmap[#] = RandomColor[])&, CZCOCOClasses ];
Map[ (CZCmap[#] = RandomColor[])&, CZPascalClasses ];
CZCmap[ _ ] = Blue;


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
TinyYoloNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/TinyYoloV2.wlnet"],"WLNet"];
CZTinyYoloOutputDecoder[ labels_, threshold_:.24 ][ netOutput_ ] := Module[{
   detections = Position[netOutput["ClassProb"]*netOutput["Objectness"],x_/;x>threshold]},k1=detections;k2=netOutput;
   Transpose[{
      Rectangle@@@Extract[netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[labels,detections[[All,2;;2]]],
      Extract[netOutput["ClassProb"]*netOutput["Objectness"], detections ]
   }]
];


(*
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
*)


(*
   Below 3 Import files were converted from Yolo
   Available form: https://github.com/pjreddie/darknet
   See licence: https://github.com/pjreddie/darknet/blob/master/LICENSE.mit
*)

yolo9000Names = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolo9000Names"],"List"];
yolo9000Graph = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolo9000Graph"],"WXF"];
yolo9000Net = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolo9000.wlnet"],"WLNet"];


CZYolo9000ClassDecoder[ startingVertex_, startingProbability_, threshold_, class_ ] := Module[ {children = Rest@VertexOutComponent[ yolo9000Graph, {startingVertex}, 1 ] },
   If[
      children=={},
      {startingVertex,startingProbability},
      If[
         Length[children]==1,
         CZYolo9000ClassDecoder[ First@children, startingProbability, class ],
         If[
            startingProbability*(SoftmaxLayer[][ class[[ children ]] ])[[ Last@Ordering[ class[[children]] ] ]] < threshold,
            { startingVertex, startingProbability },
            CZYolo9000ClassDecoder[ children[[ Last@Ordering[ class[[children]] ] ]], startingProbability*(SoftmaxLayer[][ class[[ children ]] ])[[ Last@Ordering[ class[[children]] ] ]], threshold, class ]]]]
];


CZYolo9000OutputDecoder[ labels_, threshold_:.5 ][ netOutput_ ] := Module[
   { detections = Position[ netOutput["Objectness"], x_/;x>threshold ][[All,1]] },
   Map[ Function[ {detectionBox},{
      Rectangle@@netOutput["Boxes"][[detectionBox]],
      labels[[ First@CZYolo9000ClassDecoder[ 0, netOutput["Objectness"][[ detectionBox ]], threshold, netOutput["ClassHierarchy"][[detectionBox]] ] ]],
      CZYolo9000ClassDecoder[ 0, LogisticSigmoid@netOutput["Objectness"][[ detectionBox ]], threshold, netOutput["ClassHierarchy"][[detectionBox]] ][[2]]}
   ], detections ] ];
