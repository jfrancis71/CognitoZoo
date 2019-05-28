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
   "RetinaNet", CZDetectObjectsRetinaNet[ image, FilterRules[ {opts}, Options[ CZDetectObjectsRetinaNet ] ] ] ];


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


RetinaNetR101FPNLR2Net = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/RetinaNetR101FPNLR2.wlnet"],"WLNet"];
