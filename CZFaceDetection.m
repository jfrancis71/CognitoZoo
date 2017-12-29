(* ::Package:: *)

(* Implements a multi-scale pyramid face detector (basically sliding window implemented
   convolutionally.

   Usage: HighlightImage[img,Rectangle@@@CZDetectFaces[img]]
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
Threshold->0.997
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
   CZDeleteOverlappingWindows[ Map[ {#[[1]], #[[2,1]], #[[2,2]] }&, CZMultiScaleDetectObjects[image, CZMultiScaleFaceNet, opts] ] ];


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
   HighlightImage[image,Map[{Blend[{Pink,Blue},CZGender[ImageTrim[image,#]]],Rectangle@@#}&,CZDetectFaces[image,opts]]];


(* Private Implementation Code *)

<<CZUtils.m


CZFaceNet = Import["CZModels/FaceNet.wlnet"];


scales = {493,411,342,285,238,198,165,138,115,96,80,66,55,46,38,32};
scaleNo = Length[scales];
CZMultiScaleFaceNet=NetGraph[{
   ResizeLayer[{493,493}],ResizeLayer[{411,411}],ResizeLayer[{342,342}],
   ResizeLayer[{285,285}],ResizeLayer[{238,238}],ResizeLayer[{198,198}],
   ResizeLayer[{165,165}],ResizeLayer[{138,138}],ResizeLayer[{115,115}],
   ResizeLayer[{96,96}],ResizeLayer[{80,80}],ResizeLayer[{66,66}],
   ResizeLayer[{55,55}],ResizeLayer[{46,46}],ResizeLayer[{38,38}],
   ResizeLayer[{32,32}],
   CZFaceNet,CZFaceNet,CZFaceNet,CZFaceNet,CZFaceNet,
   CZFaceNet,CZFaceNet,CZFaceNet,CZFaceNet,CZFaceNet,
   CZFaceNet,CZFaceNet,CZFaceNet,CZFaceNet,CZFaceNet,
   CZFaceNet
   },{
   1->scaleNo+1,2->scaleNo+2,3->scaleNo+3,4->scaleNo+4,5->scaleNo+5,
   6->scaleNo+6,7->scaleNo+7,8->scaleNo+8,9->scaleNo+9,10->scaleNo+10,
   11->scaleNo+11,12->scaleNo+12,13->scaleNo+13,14->scaleNo+14,15->scaleNo+15,
   16->scaleNo+16},
   "Input"->NetEncoder[{"Image",{512,512},"Grayscale"}]];


CZFaceNetDecoder[ netOutput_, threshold_ ] := Flatten[Table[
   extractPositions=Position[netOutput[[k,1]],x_/;x>threshold];
   origCoords=Map[{Extract[netOutput[[k,1]],#],4*#[[2]] + (16-4),scales[[k]]-4*#[[1]]+4-16}&,extractPositions];
   Map[{#[[1]],(512./scales[[k]])*{#[[2]]-15,#[[3]]-15},(512./scales[[k]])*{#[[2]]+16,#[[3]]+16}}&,origCoords],{k,1,16}],1]


CZFaceNetDecoder[ netOutput_, image_, threshold_ ] :=
   Map[ {#[[1]], CZTransformRectangleToImage[#[[2;;3]], image, 512] }&,  CZFaceNetDecoder[ netOutput, threshold ] ]


Options[ CZMultiScaleDetectObjects ] = {
   Threshold->0.997
};
(* Implements a sliding window object detector at multiple scales.
*)
CZMultiScaleDetectObjects[image_?ImageQ, net_, opts:OptionsPattern[] ] := (
   out = CZMultiScaleFaceNet[ImageResize[CZImagePadToSquare[image],512]];
   CZFaceNetDecoder[ out, image, OptionValue[Threshold] ]
)


GenderNet = Import["CZModels/GenderNet.wlnet"];
