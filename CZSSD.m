(* ::Package:: *)

(* Implements SSD VGG 300 on Mathematica version 11.

   SSD VGG 300 is a computer vision object detection and localisation model designed to detect
   20 object categories (e.g. people, horses, dogs etc)
   
   Usage: HighlightImage[img, CZDisplayObject /@ CZDetectObjects[img]]
   or:    CZHighlightObjects[ img ]

   Timings are around: (for the two cars on Clapham Common image)
   1.3 secs for MacBook Air
   1.1 secs Desktop CPU
   .34 secs Desktop GPU
*)

(*
   Credit:
   This implementation is based on Changan Wang's Tensorflow code:
      https://github.com/HiKapok/SSD.TensorFlow
      
   SSD VGG 300 is based on the following paper:
   https://arxiv.org/abs/1512.02325
   Title: SSD: Single Shot MultiBox Detector
   Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
   Cheng-Yang Fu, Alexander C. Berg
*)


(* Copyright Julian Francis 2018. Please see license file for details. *)


(* Public Interface Code *)


Options[ CZDetectObjects ] = {
TargetDevice->"CPU"
};
CZDetectObjects[ image_, opts:OptionsPattern[] ] :=
   CZNonMaxSuppressionPerClass@CZDeconformObjects[ CZDecodeOutput@SSDNet[ CZEncodeInput@CZConformImage[image,{300,300},"Stretch"], opts ], image, {300, 300}, "Stretch"  ]


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


(* Private Implementation Code *)


<<CZutils.m


anchorsx1 = Table[{2*x/75},{y,1,38},{x,.5,37.5}];
anchorsy1 = Table[{2*y/75},{y,.5,37.5},{x,1,38}];
anchorsw1 = {0.141,0.141,0.070};
anchorsh1 = {0.141,0.070,0.141};


anchorsx2 = Table[{4*x/75},{y,1,19},{x,.5,18.5}];
anchorsy2 = Table[{4*y/75},{y,.5,18.5},{x,1,19}];
anchorsw2 = {0.273,0.282,0.346,0.141,0.115};
anchorsh2 = {0.273,0.141,0.115,0.282,0.346};


anchorsx3 = Table[{8*x/75},{y,1,10},{x,.5,9.5}];
anchorsy3 = Table[{8*y/75},{y,.5,9.5},{x,1,10}];
anchorsw3 = {0.454,0.530,0.649,0.265,0.216};
anchorsh3 = {0.454,0.265,0.216,0.530,0.649};


anchorsx4 = Table[{16*x/75},{y,1,5},{x,.5,4.5}];
anchorsy4 = Table[{16*y/75},{y,.5,4.5},{x,1,5}];
anchorsw4 = {0.631,0.777,0.952,0.388,0.317};
anchorsh4 = {0.631,0.388,0.317,0.777,0.952};


anchorsx5 = Table[{x/3},{y,1,3},{x,.5,2.5}];
anchorsy5 = Table[{y/3},{y,.5,2.5},{x,1,3}];
anchorsw5 = {0.807,1.02,0.512};
anchorsh5 = {0.807,0.512,1.02};


anchorsx6 = Table[{x/2},{y,1,1},{x,1,1}];
anchorsy6 = Table[{y/2},{y,1,1},{x,1,1}];
anchorsw6 = {0.983,1.27,0.636};
anchorsh6 = {0.983,0.636,1.27};


SSDNet = Import[LocalCache@CloudObject["https://www.wolframcloud.com/objects/julian.w.francis/CZSSDVGG300.wlnet"],"WLNet"];


CZEncodeInput[ image_ ] :=
   (ImageData[image,Interleaving->False]*256)-{123,117,104};


CZDecodeOutput[locs_, probs_,{anchorsx_,anchorsy_,anchorsw_,anchorsh_}]:=( 
(* slocs format A{xywh}HW *)
   slocs=Partition[locs,4];
   rec=Position[probs,x_/;x>.5];
   cx=Map[#+anchorsx[[All,All,1]]&,slocs[[All,2]]*anchorsw*0.1];
   cy=Map[#+anchorsy[[All,All,1]]&,slocs[[All,1]]*anchorsh*0.1];
   w=Exp[slocs[[All,4]]*0.2]*anchorsw;
   h=Exp[slocs[[All,3]]*0.2]*anchorsh;

   MapThread[{#1,#2,Rectangle[300*{#3-#5/2,1-(#4+#6/2)},300*{#3+#5/2,1-(#4-#6/2)}]}&,{
      CZPascalClasses[[rec[[All,4]]]],
      Extract[probs,rec],
      Extract[cx,rec[[All,1;;3]]],
      Extract[cy,rec[[All,1;;3]]],
      Extract[w,rec[[All,1;;3]]],
      Extract[h,rec[[All,1;;3]]]
}]
)


CZDecodeOutput[ netOutput_ ] :=
   Join[
      CZDecodeOutput[netOutput["Locs1"],netOutput["ObjMap1"][[All,All,All,2;;21]],{anchorsx1,anchorsy1,anchorsw1,anchorsh1}],
      CZDecodeOutput[netOutput["Locs2"],netOutput["ObjMap2"][[All,All,All,2;;21]],{anchorsx2,anchorsy2,anchorsw2,anchorsh2}],
      CZDecodeOutput[netOutput["Locs3"],netOutput["ObjMap3"][[All,All,All,2;;21]],{anchorsx3,anchorsy3,anchorsw3,anchorsh3}],
      CZDecodeOutput[netOutput["Locs4"],netOutput["ObjMap4"][[All,All,All,2;;21]],{anchorsx4,anchorsy4,anchorsw4,anchorsh4}],
      CZDecodeOutput[netOutput["Locs5"],netOutput["ObjMap5"][[All,All,All,2;;21]],{anchorsx5,anchorsy5,anchorsw5,anchorsh5}],
      CZDecodeOutput[netOutput["Locs6"],netOutput["ObjMap6"][[All,All,All,2;;21]],{anchorsx6,anchorsy6,anchorsw6,anchorsh6}]
];
