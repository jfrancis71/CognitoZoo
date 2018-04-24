(* ::Package:: *)

(* Implements a multi-scale pyramid face detector (basically sliding window implemented
   convolutionally.

   Usage: HighlightImage[img,CZDetectFaces[img]]
          CZHightFaces[img] similar to above but also attempts gender recognition

   You need to ensure the following files are installed in a CZModels subfolder on your search path:
      FaceNet.wlnet, GenderNet.wlnet
   Files found in: https://drive.google.com/open?id=0Bzhe0pgVZtNUVGJJak1GWDQ3S1U 
*)
(*
   Credit:
   
   The weights for this model come from training in a neural network library called CognitoNet which is now retired.
   
   That training session used images from the Face Scrub data set:
   http: http://vintage.winklerbros.net/facescrub.html
   H.-W. Ng, S. Winkler.
   A data-driven approach to cleaning large face datasets.
   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.
*)

(* Public Interfaces *)

Options[ CZDetectFaces ] = {
   Threshold->0.997,
   TargetDevice->"CPU"
};
(* Works like FindFaces, ie returns { {{x1,y1},{x2,y2}},... }
   On the Caltech 1999 face dataset, we achieve a recognition rate of around 92% with
   an average of 14% of false positives/image.
   The Caltech dataset has 450 images where most faces are quite close to camera,
   where images are of size 896x592. Most of these images are of good quality, but some
   are challenging, eg. cartoon, significant obscuring of face or poor lighting conditions.
   Reference comparison, FindFances achieves 99.6% recognition, but 56% average false positive rate/image
*)
CZDetectFaces::usage="
CZDetectFaces[img,options] returns {{xmin1,ymin1},{xmax1,ymax1},...
Options are: Threshold

Example usage: HighlightImage[img,Rectangle@@@CZDetectFaces[img]].
";
CZDetectFaces[image_?ImageQ, opts:OptionsPattern[]] := 
   CZDetectObjects[ image, CZMultiScaleFaceNet, opts ]


Options[ CZDetectObjects ] = {
   Threshold->0.997,
   TargetDevice->"CPU"
};
CZDetectObjects[image_?ImageQ, multiScaleNet_, opts:OptionsPattern[]] := 
   CZDeleteOverlappingWindows@CZResizeObjects[ CZDecoder[ multiScaleNet[ CZEncoder@image, opts  ], OptionValue[Threshold] ], image ];


CZGender::usage = "
CZGender[image] returns a gender score ranging from 0 (most likely female) to 1 (most likely male).
";
CZGender[image_?ImageQ] :=
   (GenderNet@{pk=ColorConvert[ImageResize[image,{32,32}],"GrayScale"]//ImageData})[[1]]


CZHighlightFaces::usage = "
   CZHightFaces[image,opts] Draws bounding boxes around detected faces and attempts to determine likely gender.
   Valid option is Threshold.
";
CZHighlightFaces[image_?ImageQ,opts:OptionsPattern[]] := 
   HighlightImage[image,Map[{Blend[{Pink,Blue},CZGender[ImageTrim[image,#]]],#}&,CZDetectFaces[image,opts]]];


(* Private Implementation Code *)

<<CZUtils.m


CZFaceNet = Import["CZModels/FaceNet.wlnet"];


scales = {493,411,342,285,238,198,165,138,115,96,80,66,55,46,38,32};
CZRecognizeAtScale[ baseNet_, size_] := NetChain[ { ResizeLayer[ {size, size } ], baseNet } ];
CZBuildMultiScaleObjectNet[ baseNet_ ] := NetGraph[
   Map[ CZRecognizeAtScale[baseNet, #]&, scales ],
   Table[ k->NetPort["Output"<>ToString[k]], {k,1,16}],
   "Input"->NetEncoder[{"Image",{512,512},"Grayscale"}]
];
CZMultiScaleFaceNet = CZBuildMultiScaleObjectNet[ CZFaceNet ];


CZEncoder[ image_ ] := ImageResize[CZImagePadToSquare[image],512];


CZDecoder[ netOutput_, threshold_ ] := Flatten[Table[
   extractPositions=Position[netOutput[[k,1]],x_/;x>threshold];
   origCoords=Map[{Extract[netOutput[[k,1]],#],4*#[[2]] + (16-4),scales[[k]]-4*#[[1]]+4-16}&,extractPositions];
   Map[{#[[1]],(512./scales[[k]])*{{#[[2]]-15,#[[3]]-15},{#[[2]]+16,#[[3]]+16}}}&,origCoords],{k,1,16}],1]


CZResizeObjects[ {}, _ ] := {};
CZResizeObjects[ objects_, image_ ] :=
   Transpose[ { objects[[All,1]], CZResizeBoundingBoxes[ objects[[All,2]], image, 512 ] } ]


GenderNet = Import["CZModels/GenderNet.wlnet"];
