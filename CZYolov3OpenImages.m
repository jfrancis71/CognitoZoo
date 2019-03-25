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


yoloOpenImagesNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolov3OpenImages.wlnet"],"WLNet"];
yoloOpenImagesClasses = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/Yolov3OpenImagesClasses"],"List"];


CZOutputDecoder[ threshold_:.5 ][ output_ ] := Module[{
   probs = output["ObjMap"]*output["Classes"], detectionBoxes },
   detectionBoxes = Union@Flatten@SparseArray[UnitStep[probs-threshold]]["NonzeroPositions"][[All,1]];
   Map[ Function[{detectionBox}, {
         Rectangle@@output["Locations"][[detectionBox]],
         Map[ { yoloOpenImagesClasses[[#]], probs[[detectionBox,#]] }&, Flatten@Position[probs[[detectionBox]],x_/;x>threshold ] ] } ],
      detectionBoxes ]
];


CZNonMaxSuppression[ nmsThreshold_ ][ dets_ ] := Module[ { deletions },
(* deletions is a jagged array of form boundingBox*classDetections in bounding box where
   1 represents should be deleted
*)
   deletions =
      Table[
         Max@Table[
            If[
               boxNo!=testBoxNo&&dets[[boxNo,2,classNo,1]]==dets[[testBoxNo,2,testClassNo,1]]&&
               CZIntersectionOverUnion[dets[[boxNo,1]],dets[[testBoxNo,1]]]>nmsThreshold&&
               dets[[boxNo,2,classNo,2]]<dets[[testBoxNo,2,testClassNo,2]],1,0],
            {testBoxNo,1,Length[dets]},{testClassNo,1,Length[dets[[testBoxNo,2]]]}],
         {boxNo,1,Length[dets]},{classNo,1,Length[dets[[boxNo,2]]]}];
   DeleteCases[Delete[dets, Map[{#[[1]],2,#[[2]]}&,Position[deletions,1]]], {_,{}}]
];


CZDetectionsDeconformer[ image_Image, netDims_List, fitting_String ][ objects_ ] :=
   Transpose[ { CZDeconformRectangles[ objects[[All,1]], image, netDims, fitting ], objects[[All,2]] } ];


Options[ CZDetectObjects ] = {
   TargetDevice->"CPU",
   Threshold->.5,
   NMSIntersectionOverUnionThreshold->.45
};
CZDetectObjects[ image_Image , opts:OptionsPattern[] ] := (
   CZNonMaxSuppression[ OptionValue[ NMSIntersectionOverUnionThreshold ] ]@CZDetectionsDeconformer[ image, {608, 608}, "Fit" ]@CZOutputDecoder[ OptionValue[ Threshold ] ]@
   (yoloOpenImagesNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@CZImageConformer[{608,608},"Fit"]@image
)


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ image_Image, opts:OptionsPattern[]  ] :=
   HighlightImage[ image, CZDisplayObject/@({#[[1]],ToString@#[[2,All,1]]}&/@CZDetectObjects[ image, opts ]) ];
