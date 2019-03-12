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


CZOutputDecoder[ threshold_:.5 ][ output_ ] := (
   joint = output["ObjMap"]*output["Classes"];
   detectionBoxes = Union@Flatten@SparseArray[UnitStep[joint-threshold]]["NonzeroPositions"][[All,1]];
   Map[ {
      Rectangle@@output["Locations"][[#]],
      Transpose[{ 
         yoloOpenImagesClasses[[Flatten@Position[joint[[#]],x_/;x>threshold] ]],
         Extract[joint[[#]], Position[joint[[#]],x_/;x>threshold] ]
       }] }&, detectionBoxes ]
);


CZNonMaxSuppression[ nmsThreshold_ ][ dets_ ] := Module[ { deletions },
   deletions = Table[
      Max[
         Table[If[d!=d1&&dets[[d,2,r,1]]==dets[[d1,2,r1,1]]&&CZIntersectionOverUnion[dets[[d,1]],dets[[d1,1]]]>nmsThreshold&&dets[[d,2,r,2]]<dets[[d1,2,r1,2]],1,0],
            {d1,1,Length[dets]},{r1,1,Length[dets[[d1,2]]]}]]
      ,{d,1,Length[dets]},{r,1,Length[dets[[d,2]]]}];
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
