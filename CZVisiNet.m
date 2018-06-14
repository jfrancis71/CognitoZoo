(* ::Package:: *)

(* Public Interface *)


SyntaxInformation[NonMaximumSuppression]={"ArgumentsPattern"->{_}};
Options[ CZHighlightFaces ] = {
   TargetDevice->"CPU",
   Threshold->0.5,
   NonMaximumSuppression->"Max"
};
CZHighlightFaces[ img_Image, opts:OptionsPattern[] ] := HighlightImage[ img, CZDetectFaces[ img, opts ] ]


Options[ CZDetectFaces ] = {
   Threshold->0.5,
   TargetDevice->"CPU",
   NonMaximumSuppression->"Max"
};
CZDetectFaces[ img_Image, opts:OptionsPattern[] ] := (
   Switch[ OptionValue["NonMaximumSuppression"],
   "Max",CZNonMaxSuppression,
   "False",Identity,
   "Weighted",CZNonMaxSuppression[#,True]&]
   [CZDeconformObjects[ CZDecodeOutput[ trained[ CZConformImage[img,{640,480},"Fit"], TargetDevice->OptionValue["TargetDevice"] ], OptionValue["Threshold"] ], img, {640,480}, "Fit" ]] );


(* Private Code *)


<<CZUtils.m


trained = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/VisiNetv1"],"WLNet"];


CZDecodeOutput[ assoc_, threshold_ ] := Join[
   Map[{Extract[assoc["FaceArray1"],#],Rectangle[{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}+{37,37}]}&,Position[assoc["FaceArray1"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray2"],#],Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{65,65},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{65,65}]}&,Position[assoc["FaceArray2"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray3"],#],Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{100,100}]}&,Position[assoc["FaceArray3"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray1Offset"],#],Rectangle[{32*(#[[2]]),480-32*(#[[1]]-1)}-{37,37},{32*(#[[2]]),480-32*(#[[1]]-1)}+{37,37}]}&,Position[assoc["FaceArray1Offset"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray2Offset"],#],Rectangle[{64*(#[[2]]),480-64*(#[[1]]-1)}-{65,65},{64*(#[[2]]),480-64*(#[[1]]-1)}+{65,65}]}&,Position[assoc["FaceArray2Offset"],x_/;x>threshold]],
   Map[{Extract[assoc["FaceArray3Offset"],#],Rectangle[{64*(#[[2]]),480-64*(#[[1]]-1)}-{100,100},{64*(#[[2]]),480-64*(#[[1]]-1)}+{100,100}]}&,Position[assoc["FaceArray3Offset"],x_/;x>threshold]]
]


(*
   CloudDeploy[
      FormFunction[{"image" \[Rule]"Image"},(CZHighlightFaces[#image])&,"JPEG",AppearanceRules\[Rule] {"Title" \[Rule] "Julian's Face Detector"}],
      "CZFaceDetection"]
 *)
