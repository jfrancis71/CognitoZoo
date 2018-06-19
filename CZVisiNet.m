(* ::Package:: *)

(* Public Interface *)


<<CZUtils.m


Options[ CZDetectFaces ] = Join[{
   TargetDevice->"CPU",
   Threshold->.5
}, Options[ CZNonMaxSuppression ] ];
CZDetectFaces[ image_, opts:OptionsPattern[] ] :=
   CZNonMaxSuppression[
      CZDeconformObjects[ CZDecodeOutput[
            trained[ CZConformImage[image,{640,480},"Fit"], TargetDevice->OptionValue[ TargetDevice ] ],
            OptionValue[ Threshold ] ],
         image, {640, 480}, "Fit"  ],
      FilterRules[ {opts}, Options[ CZNonMaxSuppression ] ] ];


Options[ CZHighlightFaces ] = Options[ CZDetectFaces ];
CZHighlightFaces[ img_Image, opts:OptionsPattern[] ] := HighlightImage[ img, CZDetectFaces[ img, opts ] ]


(* Private Code *)


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
