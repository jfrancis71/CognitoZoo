(* ::Package:: *)

(*
   Implements Face Detection
   Usage: HighlightImage[img,CZDetectFaces[img]]
   or:    CZHightFaces[img]
   
Timing:
   All Timings for a 640x480 image preimported. GPU is Quadro K4200

   CZDetectFaces[img]//AbsoluteTiming
   1.1 secs
   
   CZDetectFaces[img,Detail\[Rule]"VGA"]//AbsoluteTiming
   0.19 secs
   
   CZDetectFaces[img,TargetDevice\[Rule]"GPU"]//AbsoluteTiming
   0.24 secs
   
   CZDetectFaces[img,TargetDevice\[Rule]"GPU",Detail\[Rule]"VGA"]//AbsoluteTiming
   0.04 secs
   
   Performance on gender detection:
      7% error tested on a BBC Question Time video (with detail\[Rule]"VGA")
      5% with adding CZTakeWeightedRectangles
*)


(* Credit

   Training net comes from ./Training/VisiNet.m
   That training session used images from the Face Scrub data set:
   http: http://vintage.winklerbros.net/facescrub.html
   H.-W. Ng, S. Winkler.
   A data-driven approach to cleaning large face datasets.
   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.
*)


(* Public Interface *)


<<CZUtils.m


SyntaxInformation[ Detail ]= {"ArgumentsPattern"->{_}};
SyntaxInformation[ GenderDetection ]= {"ArgumentsPattern"->{_}};
Options[ CZDetectFaces ] = Join[{
   TargetDevice->"CPU",
   Threshold->.5,
   Detail->"XGA",
   GenderDetection->False
}, Options[ CZNonMaxSuppression ] ];


CZDetectFaces[ image_Image, opts:OptionsPattern[] ] :=
   Function[detections,If[OptionValue[GenderDetection],Map[{#[[1]],CZGenderClassify[#[[3,1]]]}&,detections],detections[[All,1]]]]@
   CZNonMaxSuppression[ FilterRules[ {opts}, Options[ CZNonMaxSuppression ] ] ]@If[OptionValue[Detail]=="VGA",
      CZObjectsDeconformer[ image, {640, 480}, "Fit" ]@(CZVGADetectFaces[#,FilterRules[{opts},Options[ CZVGADetectFaces ] ] ]&)@CZImageConformer[{640,480},"Fit"]@image,
      CZObjectsDeconformer[ image, {1280, 960}, "Fit" ]@(CZXGADetectFaces[#,FilterRules[{opts},Options[ CZXGADetectFaces ] ] ]&)@CZImageConformer[{1280,960},"Fit"]@image
   ]


Options[ CZHighlightFaces ] = Options[ CZDetectFaces ];
CZHighlightFaces[ img_Image, opts:OptionsPattern[] ] := HighlightImage[
   img,Reverse/@(*We want to set the colour before drawing the rectangle (not after!*)
   If[ OptionValue[GenderDetection],
      # /. "Male"->Blue /. "Female" ->Pink,
      # ]&
      @CZDetectFaces[ img, opts ]
]


(* Private Code *)


Options[ CZVGADetectFaces ] = {
   TargetDevice->"CPU",
   Threshold->.5
};
CZVGADetectFaces[ image_Image, opts:OptionsPattern[] ] :=
   CZOutputDecoder[ OptionValue[ Threshold ], {640, 480} ]@(CZVisiNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@image;


CZMapAt[_,{},_]:={};
CZMapAt[f_,list_,spec_]:=MapAt[f,list,spec];


Options[ CZXGADetectFaces ] = {
   TargetDevice->"CPU",
   Threshold->.5
};
CZXGADetectFaces[ image_Image, opts:OptionsPattern[] ] := Join[
   CZObjectsDeconformer[ image, {640, 480}, "Fit" ]@CZVGADetectFaces[ CZImageConformer[{640,480},"Fit"]@image, opts ],
CZOutputDecoder[ OptionValue[ Threshold ], {1280,960} ]@(CZXGAVisiNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@image
];


CZVisiNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/CZVisiNetVGAFaceScrubv2.wlnet"],"WLNet"];
CZXGAVisiNet=NetReplacePart[CZVisiNet,"Input"->NetEncoder[{"Image",{1280,960},ColorSpace->"RGB"}]];


CZGenderClassify[ maleness_ ] := If[ maleness >= .5, "Male", "Female" ]


CZOutputDecoder[ threshold_, { width_, height_ } ][ netOutput_ ] :=Join[
   Map[{
         Rectangle[{32*(#[[2]]-.5),height-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),height-32*(#[[1]]-.5)}+{37,37}],
         Extract[netOutput["FaceArray1"],#],
         {Extract[netOutput["GenderArray1"],#]}
         }&,
      Position[netOutput["FaceArray1"],x_/;x>threshold]],
   Map[{
         Rectangle[{64*(#[[2]]-.5),height-64*(#[[1]]-.5)}-{65,65},{64*(#[[2]]-.5),height-64*(#[[1]]-.5)}+{65,65}],
         Extract[netOutput["FaceArray2"],#],
         {Extract[netOutput["GenderArray2"],#]}
         }&,
      Position[netOutput["FaceArray2"],x_/;x>threshold]],
   Map[{
         Rectangle[{64*(#[[2]]-.5),height-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),height-64*(#[[1]]-.5)}+{100,100}],
         Extract[netOutput["FaceArray3"],#],
         {Extract[netOutput["GenderArray3"],#]}
         }&,
      Position[netOutput["FaceArray3"],x_/;x>threshold]],
   Map[{
         Rectangle[{32*(#[[2]]),height-32*(#[[1]]-1)}-{37,37},{32*(#[[2]]),height-32*(#[[1]]-1)}+{37,37}],
         Extract[netOutput["FaceArray1Offset"],#],
         {Extract[netOutput["GenderArray1Offset"],#]}
         }&,
      Position[netOutput["FaceArray1Offset"],x_/;x>threshold]],
   Map[{
         Rectangle[{64*(#[[2]]),height-64*(#[[1]]-1)}-{65,65},{64*(#[[2]]),height-64*(#[[1]]-1)}+{65,65}],
         Extract[netOutput["FaceArray2Offset"],#],
         {Extract[netOutput["GenderArray2Offset"],#]}
         }&,
      Position[netOutput["FaceArray2Offset"],x_/;x>threshold]],
   Map[{
         Rectangle[{64*(#[[2]]),height-64*(#[[1]]-1)}-{100,100},{64*(#[[2]]),height-64*(#[[1]]-1)}+{100,100}],
         Extract[netOutput["FaceArray3Offset"],#],
         {Extract[netOutput["GenderArray3Offset"],#]}
         }&,
      Position[netOutput["FaceArray3Offset"],x_/;x>threshold]]
];


(*
   CloudDeploy[
      FormFunction[{"image" \[Rule]"Image"},(CZHighlightFaces[#image])&,"JPEG",AppearanceRules\[Rule] {"Title" \[Rule] "Julian's Face Detector"}],
      "CZFaceDetection"]
 *)
