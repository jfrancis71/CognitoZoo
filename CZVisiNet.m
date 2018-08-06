(* ::Package:: *)

(*
   Implements Face Detection
   Usage: HighlightImage[img,CZDetectFaces[img]]
   or:    CZHightFaces[img]
   
Timing:
   All Timings for a 640x480 image preimported. GPU is Quadro K4200

   CZDetectFaces[img]//AbsoluteTiming
   1.6 secs
   
   CZDetectFaces[img,Detail\[Rule]"VGA"]//AbsoluteTiming
   0.19 secs
   
   CZDetectFaces[img,TargetDevice\[Rule]"GPU"]//AbsoluteTiming
   1.52 secs
   
   CZDetectFaces[img,TargetDevice\[Rule]"GPU",Detail\[Rule]"VGA"]//AbsoluteTiming
   0.04 secs
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
   Function[detections,If[OptionValue[GenderDetection],Map[{CZGenderClassify[#[[2,1]]],#[[3]]}&,detections],detections[[All,3]]]]@
   CZNonMaxSuppressionPerClass[ FilterRules[ {opts}, Options[ CZNonMaxSuppressionPerClass ] ] ]@If[OptionValue[Detail]=="VGA",
      CZObjectsDeconformer[ image, {640, 480}, "Fit" ]@(CZVGADetectFaces[#,FilterRules[{opts},Options[ CZVGADetectFaces ] ] ]&)@CZImageConformer[{640,480},"Fit"]@image,
      CZObjectsDeconformer[ image, {1280, 960}, "Fit" ]@(CZXGADetectFaces[#,FilterRules[{opts},Options[ CZXGADetectFaces ] ] ]&)@CZImageConformer[{1280,960},"Fit"]@image
   ]   


Options[ CZHighlightFaces ] = Options[ CZDetectFaces ];
CZHighlightFaces[ img_Image, opts:OptionsPattern[] ] := HighlightImage[
   img,
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
   CZOutputDecoder[ OptionValue[ Threshold ] ]@(CZVisiNet[ #, TargetDevice->OptionValue[ TargetDevice ] ]&)@image;


CZMapAt[_,{},_]:={};
CZMapAt[f_,list_,spec_]:=MapAt[f,list,spec]


Options[ CZXGADetectFaces ] = {
   TargetDevice->"CPU",
   Threshold->.5
};
CZXGADetectFaces[ image_Image, opts:OptionsPattern[] ] := Join[
   CZObjectsDeconformer[ image, {640, 480}, "Fit" ]@CZVGADetectFaces[ CZImageConformer[{640,480},"Fit"]@image, opts ],
   Flatten[MapThread[
      Function[{objects,offset},CZMapAt[(#+offset)&,objects,{All,4,All}]],{
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


CZVisiNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/CZVisiNetv2.wlnet"],"WLNet"];


CZGenderClassify[ maleness_ ] := If[ maleness >= .5, "Male", "Female" ]


CZOutputDecoder[ threshold_ ] := Function[ { assoc }, Join[
   Map[{
         "Face",
         Extract[assoc["FaceArray1"],#],
         {Extract[assoc["GenderArray1"],#]},
         Rectangle[{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}+{37,37}]}&,
      Position[assoc["FaceArray1"],x_/;x>threshold]],
   Map[{
         "Face",
         Extract[assoc["FaceArray2"],#],
         {Extract[assoc["GenderArray2"],#]},
         Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{65,65},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{65,65}]}&,
      Position[assoc["FaceArray2"],x_/;x>threshold]],
   Map[{
         "Face",
         Extract[assoc["FaceArray3"],#],
         {Extract[assoc["GenderArray3"],#]},
         Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{100,100}]}&,
      Position[assoc["FaceArray3"],x_/;x>threshold]],
   Map[{
         "Face",
         Extract[assoc["FaceArray1Offset"],#],
         {Extract[assoc["GenderArray1Offset"],#]},
         Rectangle[{32*(#[[2]]),480-32*(#[[1]]-1)}-{37,37},{32*(#[[2]]),480-32*(#[[1]]-1)}+{37,37}]}&,
      Position[assoc["FaceArray1Offset"],x_/;x>threshold]],
   Map[{
         "Face",
         Extract[assoc["FaceArray2Offset"],#],
         {Extract[assoc["GenderArray2Offset"],#]},
         Rectangle[{64*(#[[2]]),480-64*(#[[1]]-1)}-{65,65},{64*(#[[2]]),480-64*(#[[1]]-1)}+{65,65}]}&,
      Position[assoc["FaceArray2Offset"],x_/;x>threshold]],
   Map[{
         "Face",
         Extract[assoc["FaceArray3Offset"],#],
         {Extract[assoc["GenderArray3Offset"],#]},
         Rectangle[{64*(#[[2]]),480-64*(#[[1]]-1)}-{100,100},{64*(#[[2]]),480-64*(#[[1]]-1)}+{100,100}]}&,
      Position[assoc["FaceArray3Offset"],x_/;x>threshold]]
] ]


(*
   CloudDeploy[
      FormFunction[{"image" \[Rule]"Image"},(CZHighlightFaces[#image])&,"JPEG",AppearanceRules\[Rule] {"Title" \[Rule] "Julian's Face Detector"}],
      "CZFaceDetection"]
 *)
