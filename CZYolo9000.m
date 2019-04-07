(* ::Package:: *)

Options[ CZDetectObjects ] = {
   TargetDevice->"CPU",
   Threshold->.5,
   NMSIntersectionOverUnionThreshold->.45
};
CZDetectObjects[ image_Image , opts:OptionsPattern[] ] := (
   CZNonMaxSuppression[ OptionValue[ NMSIntersectionOverUnionThreshold ] ]@CZObjectsDeconformer[ image, {544, 544}, "Fit" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (yolo9000Init[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@CZImageConformer[{544,544},"Fit", Padding->0.5]@image
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


Clear[classDecoder]


classDecoder[ startingVertex_, startingProbability_, threshold_, class_ ] := Module[ {children = Rest@VertexOutComponent[ yolo9000Graph, {startingVertex}, 1 ] },
   If[
      children=={},
      {startingVertex,startingProbability},
      If[
         Length[children]==1,
         classDecoder[ First@children, startingProbability, class ],
         If[
            startingProbability*(SoftmaxLayer[][ class[[ children ]] ])[[ Last@Ordering[ class[[children]] ] ]] < threshold,
            { startingVertex, startingProbability },
            classDecoder[ children[[ Last@Ordering[ class[[children]] ] ]], startingProbability*(SoftmaxLayer[][ class[[ children ]] ])[[ Last@Ordering[ class[[children]] ] ]], threshold, class ]]]]
];


CZOutputDecoder[ threshold_:.5 ][ netOutput_ ] := Module[
   { detections = Position[ netOutput["Objectness"], x_/;x>0 ][[All,1]] },dets1=detections;nets1=netOutput;
   Map[ Function[ {detectionBox},{
      Rectangle@@netOutput["Boxes"][[detectionBox]],
      yolo9000Names[[ First@classDecoder[ 0, LogisticSigmoid@netOutput["Objectness"][[ detectionBox ]], threshold, netOutput["ClassHierarchy"][[detectionBox]] ] ]],
      classDecoder[ 0, LogisticSigmoid@netOutput["Objectness"][[ detectionBox ]], threshold, netOutput["ClassHierarchy"][[detectionBox]] ][[2]]}
   ], detections ] ];
