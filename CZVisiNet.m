(* ::Package:: *)

(* Public Interface *)


<<CZUtils.m


SyntaxInformation[ Detail ]= {"ArgumentsPattern"->{_}};
Options[ CZDetectFaces ] = Join[{
   TargetDevice->"CPU",
   Threshold->.5,
   Detail->"XGA"
}, Options[ CZNonMaxSuppression ] ];


CZDetectFaces[ image_Image, opts:OptionsPattern[] ] :=
   CZNonMaxSuppression[FilterRules[ {opts}, Options[ CZNonMaxSuppression ] ] ]@
   If[OptionValue[Detail]=="VGA",
      CZObjectsDeconformer[ image, {640, 480}, "Fit" ]@(CZVGADetectFaces[#,FilterRules[{opts},Options[ CZVGADetectFaces ] ] ]&)@CZImageConformer[{640,480},"Fit"]@image,
      CZObjectsDeconformer[ image, {1280, 960}, "Fit" ]@(CZXGADetectFaces[#,FilterRules[{opts},Options[ CZXGADetectFaces ] ] ]&)@CZImageConformer[{1280,960},"Fit"]@image
   ]   


Options[ CZHighlightFaces ] = Options[ CZDetectFaces ];
CZHighlightFaces[ img_Image, opts:OptionsPattern[] ] := HighlightImage[ img, CZDetectFaces[ img, opts ] ]


(* Private Code *)


Options[ CZVGADetectFaces ] = {
   TargetDevice->"CPU",
   Threshold->.5
};
CZVGADetectFaces[ image_Image, opts:OptionsPattern[] ] :=
   CZOutputDecoder[ OptionValue[ Threshold ] ]@(trained[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@image;


CZMapAt[_,{},_]:={};
CZMapAt[f_,list_,spec_]:=MapAt[f,list,spec]


Options[ CZXGADetectFaces ] = {
   TargetDevice->"CPU",
   Threshold->.5
};
CZXGADetectFaces[ image_Image, opts:OptionsPattern[] ] := Join[
   CZObjectsDeconformer[ image, {640, 480}, "Fit" ]@CZVGADetectFaces[ CZImageConformer[{640,480},"Fit"]@image, opts ],
   Flatten[MapThread[
      Function[{objects,offset},CZMapAt[(#+offset)&,objects,{All,2,All}]],{
      Map[CZVGADetectFaces,ImageTrim[image,{
      {{1,1},{640,480}},(*bottom left*)
      {{1,481},{641,960}},(*top left*)
      {{641,1},{1280,480}},(*bottom right*)
     {{641,481},{1280,960}},(*top right*)
     {{320,240},{960,720}},(*centre*)
     {{1,240},{640,720}},(*centre left*)
     {{641,240},{1280,720}}(*centre right*)
   }]],
   {{0,0},{0,480},{640,0},{640,480},{320,240},{0,240},{640,240}}
   }],1]
];


trained = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/VisiNetv1"],"WLNet"];


CZOutputDecoder[ threshold_ ] := Function[ { assoc }, Join[
   Map[{Extract[assoc["FaceArray1"],#],Rectangle[{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}+{37,37}]}&,Position[assoc["FaceArray1"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray2"],#],Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{65,65},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{65,65}]}&,Position[assoc["FaceArray2"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray3"],#],Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{100,100}]}&,Position[assoc["FaceArray3"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray1Offset"],#],Rectangle[{32*(#[[2]]),480-32*(#[[1]]-1)}-{37,37},{32*(#[[2]]),480-32*(#[[1]]-1)}+{37,37}]}&,Position[assoc["FaceArray1Offset"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray2Offset"],#],Rectangle[{64*(#[[2]]),480-64*(#[[1]]-1)}-{65,65},{64*(#[[2]]),480-64*(#[[1]]-1)}+{65,65}]}&,Position[assoc["FaceArray2Offset"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray3Offset"],#],Rectangle[{64*(#[[2]]),480-64*(#[[1]]-1)}-{100,100},{64*(#[[2]]),480-64*(#[[1]]-1)}+{100,100}]}&,Position[assoc["FaceArray3Offset"],x_/;x>threshold]]
] ]


(*
   CloudDeploy[
      FormFunction[{"image" \[Rule]"Image"},(CZHighlightFaces[#image])&,"JPEG",AppearanceRules\[Rule] {"Title" \[Rule] "Julian's Face Detector"}],
      "CZFaceDetection"]
 *)
