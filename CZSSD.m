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
   

   You need to ensure the following files are installed in a CZModels subfolder on your search path:
      SSDVGG300.wlnet
   Files found in: https://drive.google.com/open?id=0Bzhe0pgVZtNUVGJJak1GWDQ3S1U 
*)

(*
   Credit:
   
   Paul Balanca's Tensorflow implementation was used as a reference implementation:
      https://github.com/balancap/SSD-Tensorflow
      
   SSD VGG 300 is based on the following paper:
   https://arxiv.org/abs/1512.02325
   Title: SSD: Single Shot MultiBox Detector
   Authors: Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
   Cheng-Yang Fu, Alexander C. Berg
*)
(*
The following copyright notice applies to the neural network weight file only (ssd.hdf )
   which has been converted from Paul Balanca's TensorFlow checkpoint file.

# Copyright 2016 Paul Balanca. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*)


(* Copyright Julian Francis 2018. Please see license file for details. *)


(* Public Interface Code *)


Options[ CZDetectObjects ] = {
TargetDevice->"CPU"
};
CZDetectObjects[ image_, opts:OptionsPattern[] ] :=
   CZPerClassNonMaxSuppression@CZDeconformObjects[ CZDecoder@SSDNet[ CZEncoder@image, opts ], image, {300, 300}, "Stretch"  ]


Options[ CZHighlightObjects ] = Options[ CZDetectObjects ];
CZHighlightObjects[ img_, opts:OptionsPattern[] ] := (
   HighlightImage[img,
      CZDisplayObject /@ CZDetectObjects[ img, opts ]]
)


(* Private Implementation Code *)


<<CZutils.m


anchorsx1 = Table[{2*x/75},{y,1,38},{x,.5,37.5}];
anchorsy1 = Table[{2*y/75},{y,.5,37.5},{x,1,38}];
anchorsw1 = {0.070,0.102,0.098,0.049};
anchorsh1 = {0.070,0.102,0.049,0.098};


anchorsx2 = Table[{4*x/75},{y,1,19},{x,.5,18.5}];
anchorsy2 = Table[{4*y/75},{y,.5,18.5},{x,1,19}];
anchorsw2 = {0.150,0.222,0.212,0.106,0.259,0.086};
anchorsh2 = {0.150,0.222,0.106,0.212,0.086,0.259};


anchorsx3 = Table[{8*x/75},{y,1,10},{x,.5,9.5}];
anchorsy3 = Table[{8*y/75},{y,.5,9.5},{x,1,10}];
anchorsw3 = {0.330,0.410,0.466,0.233,0.571,0.190};
anchorsh3 = {0.330,0.410,0.233,0.466,0.190,0.571};


anchorsx4 = Table[{16*x/75},{y,1,5},{x,.5,4.5}];
anchorsy4 = Table[{16*y/75},{y,.5,4.5},{x,1,5}];
anchorsw4 = {0.509,0.593,0.721,0.360,0.883,0.294};
anchorsh4 = {0.509,0.593,0.360,0.721,0.294,0.883};


anchorsx5 = Table[{x/3},{y,1,3},{x,.5,2.5}];
anchorsy5 = Table[{y/3},{y,.5,2.5},{x,1,3}];
anchorsw5 = {0.689,0.774,0.975,0.487};
anchorsh5 = {0.689,0.774,0.487,0.975};


anchorsx6 = Table[{x/2},{y,1,1},{x,1,1}];
anchorsy6 = Table[{y/2},{y,1,1},{x,1,1}];
anchorsw6 = {0.870,0.955,1.230,0.615};
anchorsh6 = {0.870,0.955,0.615,1.230};


SSDNet=Import["CZModels/SSDVGG300.wlnet"];


CZEncoder[ image_ ] :=
   (ImageData[ImageResize[image,{300,300}],Interleaving->False]*256)-{123,117,104};


CZDecoder[locs_, probs_,{anchorsx_,anchorsy_,anchorsw_,anchorsh_}]:=( 
(* slocs format A{xywh}HW *)
   slocs=Partition[locs,4];
   rec=Position[probs,x_/;x>.5];
   cx=Map[#+anchorsx[[All,All,1]]&,slocs[[All,1]]*anchorsw*0.1];
   cy=Map[#+anchorsy[[All,All,1]]&,slocs[[All,2]]*anchorsh*0.1];
   w=Exp[slocs[[All,3]]*0.2]*anchorsw;
   h=Exp[slocs[[All,4]]*0.2]*anchorsh;

   MapThread[{#1,#2,Rectangle[300*{#3-#5/2,1-(#4+#6/2)},300*{#3+#5/2,1-(#4-#6/2)}]}&,{
      CZPascalClasses[[rec[[All,4]]]],
      Extract[probs,rec],
      Extract[cx,rec[[All,1;;3]]],
      Extract[cy,rec[[All,1;;3]]],
      Extract[w,rec[[All,1;;3]]],
      Extract[h,rec[[All,1;;3]]]
}]
)


CZDecoder[ netOutput_ ] :=
   Join[
      CZDecoder[netOutput["Locs1"],netOutput["ObjMap1"][[All,All,All,2;;21]],{anchorsx1,anchorsy1,anchorsw1,anchorsh1}],
      CZDecoder[netOutput["Locs2"],netOutput["ObjMap2"][[All,All,All,2;;21]],{anchorsx2,anchorsy2,anchorsw2,anchorsh2}],
      CZDecoder[netOutput["Locs3"],netOutput["ObjMap3"][[All,All,All,2;;21]],{anchorsx3,anchorsy3,anchorsw3,anchorsh3}],
      CZDecoder[netOutput["Locs4"],netOutput["ObjMap4"][[All,All,All,2;;21]],{anchorsx4,anchorsy4,anchorsw4,anchorsh4}],
      CZDecoder[netOutput["Locs5"],netOutput["ObjMap5"][[All,All,All,2;;21]],{anchorsx5,anchorsy5,anchorsw5,anchorsh5}],
      CZDecoder[netOutput["Locs6"],netOutput["ObjMap6"][[All,All,All,2;;21]],{anchorsx6,anchorsy6,anchorsw6,anchorsh6}]
];
