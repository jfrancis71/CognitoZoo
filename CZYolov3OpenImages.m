(* ::Package:: *)

(*
   Mathematica implementation of Yolo v3 Open Images
   
   Usage: CZHighlightObjects[ image ]
   
   Neural net file yoloOpenImagesNet is converted from https://pjreddie.com/darknet/yolo/
   License: https://github.com/pjreddie/darknet/blob/master/LICENSE.mit
   
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
*)


<<CZUtils.m

(*
   Below 2 Import files were converted from Yolo
   Available form: https://github.com/pjreddie/darknet
   See licence: https://github.com/pjreddie/darknet/blob/master/LICENSE.mit
*)

yoloOpenImagesNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolov3OpenImages.wlnet"],"WLNet"];
yoloOpenImagesClasses = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolov3OpenImagesClasses"],"List"];


CZOutputDecoderYoloOpenImages[ threshold_:.5 ][ output_ ] := Module[{
   probs = output["Objectness"]*output["ClassProb"], detectionBoxes },
   detectionBoxes = Union@Flatten@SparseArray[UnitStep[probs-threshold]]["NonzeroPositions"][[All,1]];
   Map[ Function[{detectionBox}, {
         Rectangle@@output["Boxes"][[detectionBox]],
         Map[ { yoloOpenImagesClasses[[#]], probs[[detectionBox,#]] }&, Flatten@Position[probs[[detectionBox]],x_/;x>threshold ] ] } ],
      detectionBoxes ]
];


CZNonMaxSuppressionYoloOpenImages[ maxOverlapFraction_ ][ dets_ ] :=
   DeleteCases[ Function[detection, {detection[[1]],
      Function[overlapBoxLabels,
         Select[detection[[2]],#[[2]]>Max@Extract[overlapBoxLabels[[All,All,2]],Position[overlapBoxLabels[[All,All,1]],#[[1]]]]&]]
      [ Select[dets,(CZIntersectionOverUnion[detection[[1]],#[[1]]]>maxOverlapFraction&&!(detection[[1]]===#[[1]]))&][[All,2]] ] }]/@dets,
      {_,{}}];


CZDetectionsDeconformerYoloOpenImages[ image_Image, netDims_List, fitting_String ][ objects_ ] :=
   Transpose[ { CZDeconformRectangles[ objects[[All,1]], image, netDims, fitting ], objects[[All,2]] } ];


CZFilterClassesYoloOpenImages[ All ][ detections_ ] := detections;
CZFilterClassesYoloOpenImages[ classes_ ][ detections_ ] :=
   DeleteCases[ {#[[1]], Select[#[[2]], Function[det, MemberQ[classes, det[[1]] ]] ]}&/@detections, { _, {} } ];


SyntaxInformation[ DetectionClasses ]= {"ArgumentsPattern"->{_}};
Options[ CZDetectYoloOpenImages ] = {
   TargetDevice->"CPU",
   AcceptanceThreshold->.5,
   MaxOverlapFraction->.45,
   DetectionClasses->All
};
CZDetectYoloOpenImages[ image_Image , opts:OptionsPattern[] ] := (
   CZNonMaxSuppression[ OptionValue[ MaxOverlapFraction ] ]@CZDetectionsDeconformerYoloOpenImages[ image, {608, 608}, "Fit" ]@CZFilterClassesYoloOpenImages[ OptionValue[ DetectionClasses ] ]@CZOutputDecoderYoloOpenImages[ OptionValue[ AcceptanceThreshold ] ]@
   (yoloOpenImagesNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@CZImageConformer[ {608,608}, "Fit", Padding->0.5 ]@image
)


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightYoloOpenImages[ image_Image, opts:OptionsPattern[]  ] :=
   HighlightImage[ image, CZDisplayObject/@({#[[1]],ToString@#[[2,All,1]]}&/@CZDetectYoloOpenImages[ image, opts ]) ];
