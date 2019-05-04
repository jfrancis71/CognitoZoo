(* ::Package:: *)

Options[ CZDetectObjects ] = Join[{
   TargetDevice->"CPU",
   Threshold->.6,
   NMSIntersectionOverUnionThreshold->.45 (* This is the Wei Liu default setting for this implementation *)
}, Options[ CZNonMaxSuppressionPerClass ] ];
CZDetectObjects[ img_Image, opts:OptionsPattern[] ] :=
   CZNonMaxSuppressionPerClass[FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@
   CZObjectsDeconformer[ img, {1152, 896}, "Fit" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (k=(RetinaNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@
   CZImageConformer[{1152,896},"Fit"]@img);


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


<<CZUtils.m


RetinaNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/RetinaNetR101FPNLR2.wlnet"],"WLNet"];


CZCOCOClasses = {"person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear",
"zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
"kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup",
"fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut",
"cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
"cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
"hair drier","toothbrush"};


(* Private Implementation Code *)


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[{
   detections = Position[Normal@netOutput["ClassProb"],x_/;x>threshold]},
   Transpose[{
      Rectangle@@@Extract[Normal@netOutput["Boxes"],detections[[All,1;;1]]],
      Extract[CZCOCOClasses,detections[[All,2;;2]]],
      Extract[Normal@netOutput["ClassProb"], detections ]
   }]
];
