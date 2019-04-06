(* ::Package:: *)

Options[ CZDetectObjects ] = {
   TargetDevice->"CPU",
   Threshold->.5,
   NMSIntersectionOverUnionThreshold->.45
};
CZDetectObjects[ image_Image , opts:OptionsPattern[] ] := (
   CZNonMaxSuppression[ OptionValue[ NMSIntersectionOverUnionThreshold ] ]@CZObjectsDeconformer[ image, {544, 544}, "Fit" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (yolo9000Init[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@CZImageConformer[{544,544},"Fit"]@image
)


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


<<CZUtils.m


yolo9000Hierarchy = Import["/Users/julian/yolov3/darknet/data/9k.tree"];


yolo9000Graph = Table[(yolo9000Hierarchy[[k,2]]+1)->k,{k,1,9418}];


yolo9000Names = Import["~/yolov3/darknet/data/9k.names","List"];


classDecoder[ startingVertex_, class_ ] := Module[ {children = Rest@VertexOutComponent[ yolo9000Graph, {startingVertex}, 1 ] },
   If[children=={},startingVertex,classDecoder[ children[[ Last@Ordering[ class[[children]] ] ]], class ]]
];


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[
   { detections = Position[ netOutput["Objectness"], x_/;x>0 ][[All,1]] },
   Map[ Function[ {detectionBox},{
      Rectangle@@netOutput["Boxes"][[detectionBox]],
      yolo9000Names[[ classDecoder[ 0, netOutput["ClassHierarchy"][[detectionBox]] ] ]],
      netOutput["Objectness"][[ detectionBox ]]}
   ], detections ] ];
