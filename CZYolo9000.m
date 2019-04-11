(* ::Package:: *)

(*
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
*)

Options[ CZDetectObjects ] = {
   TargetDevice->"CPU",
   Threshold->.5,
   NMSIntersectionOverUnionThreshold->.45
};
CZDetectObjects[ image_Image , opts:OptionsPattern[] ] := (
   CZNonMaxSuppression[ OptionValue[ NMSIntersectionOverUnionThreshold ] ]@CZObjectsDeconformer[ image, {544, 544}, "Fit" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (yolo9000Net[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@CZImageConformer[{544,544},"Fit", Padding->0.5]@image
)


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_Image, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


<<CZUtils.m

(*
   Below 3 Import files were converted from Yolo
   Available form: https://github.com/pjreddie/darknet
   See licence: https://github.com/pjreddie/darknet/blob/master/LICENSE.mit
*)

yolo9000Names = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolo9000Names"],"List"];
yolo9000Graph = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolo9000Graph"],"WXF"];
yolo9000Net = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolo9000.wlnet"],"WLNet"];


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
   { detections = Position[ netOutput["Objectness"], x_/;x>threshold ][[All,1]] },
   Map[ Function[ {detectionBox},{
      Rectangle@@netOutput["Boxes"][[detectionBox]],
      yolo9000Names[[ First@classDecoder[ 0, netOutput["Objectness"][[ detectionBox ]], threshold, netOutput["ClassHierarchy"][[detectionBox]] ] ]],
      classDecoder[ 0, LogisticSigmoid@netOutput["Objectness"][[ detectionBox ]], threshold, netOutput["ClassHierarchy"][[detectionBox]] ][[2]]}
   ], detections ] ];
