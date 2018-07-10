(* ::Package:: *)

(*
   Training file for a single pass face detector.
   Input image should be of size VGA.

   Performance: CZHighlightFaces[img,Threshold\[Rule]0.5]
      takes 0.16 secs.
*)


(*
   H.-W. Ng, S. Winkler.
   A data-driven approach to cleaning large face datasets.
   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.

   http://vintage.winklerbros.net/facescrub.html
*)


size[ face_ ] := (face[[2,1]]-face[[1,1]])


CZReplacePart[array_,rules_] := ReplacePart[ array, Select[ rules, #[[1,1]] > 0 && #[[1,2]] > 0 && #[[1,3]] > 0 & ] ]


CZCentroidsToArray[ centroids_, inputDims_, arrayDims_, stride_, offset_ ] :=
   CZReplacePart[ ConstantArray[ 0, arrayDims ], kt=Map[{Ceiling[(1+inputDims[[2]]-#[[2]])/stride+offset],(1+Floor[(#[[1]]-1)/stride-offset])}->1&,centroids] ]


(* Note we're encoding from the top so 1st row is 417-480 inclusive *)
(* offset has the grid shifted to the right and up, so grid cell 8,1 (at level 2) represents 17-32, 17-32 *)
CZEncodeTarget[ faces_, offset_ ] := {
   CZCentroidsToArray[ RegionCentroid/@Select[faces,size[#]<108&], { 640, 480 }, { 15, 20 }, 32, offset ],
   CZCentroidsToArray[ RegionCentroid/@Select[faces,size[#]>=108&&size[#]<=155&], { 640, 480 }, { 8, 10 }, 64, offset ],
   CZCentroidsToArray[ RegionCentroid/@Select[faces,size[#]>155&], { 640, 480 }, { 8, 10 }, 64, offset ]
};


mfiles1=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages1\\*.jpg"]];
mfaces1=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces1.mx"];
mfiles2=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages2\\*.jpg"]];
mfaces2=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces2.mx"];
mfiles3=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages3\\*.jpg"]];
mfaces3=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces3.mx"];
mfiles4=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages4\\*.jpg"]];
mfaces4=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces4.mx"];
mfiles5=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages5\\*.jpg"]];
mfaces5=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces5.mx"];
mfiles6=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages6\\*.jpg"]];
mfaces6=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces6.mx"];
mfiles=Join[mfiles1,mfiles2,mfiles3,mfiles4,mfiles5,mfiles6];
mfaces=Join[mfaces1,mfaces2,mfaces3,mfaces4,mfaces5,mfaces6];


ffiles1=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\ActressImages1\\*.jpg"]];
ffaces1=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\DLibFaces1.mx"];
ffiles2=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\ActressImages2\\*.jpg"]];
ffaces2=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\DLibFaces2.mx"];
ffiles3=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\ActressImages3\\*.jpg"]];
ffaces3=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\DLibFaces3.mx"];
ffiles4=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\ActressImages4\\*.jpg"]];
ffaces4=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\DLibFaces4.mx"];
ffiles5=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\ActressImages5\\*.jpg"]];
ffaces5=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\DLibFaces5.mx"];
ffiles6=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\ActressImages6\\*.jpg"]];
ffaces6=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActressImages\\VGA\\DLibFaces6.mx"];
ffiles=Join[ffiles1,ffiles2,ffiles3,ffiles4,ffiles5,ffiles6];
ffaces=Join[ffaces1,ffaces2,ffaces3,ffaces4,ffaces5,ffaces6];


files=Join[mfiles,ffiles];
faces=Join[mfaces,ffaces];


(* rnds = RandomSample@Range[ Length[ faces ] ]; *)


(* Export["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\VGATrainingRandomisation.mx",rnds]; *)


rnds = Import["c:\\Users\\julian\\ImageDataSets\\FaceScrub\\VGATrainingRandomisation.mx"];


ds = Dataset[
   Table[
      Association[
         "Input"->files[[k]],
         "FaceArray1"->CZEncodeTarget[faces[[k]],0.0][[1]],
         "FaceArray2"->CZEncodeTarget[faces[[k]],0.0][[2]],
         "FaceArray3"->CZEncodeTarget[faces[[k]],0.0][[3]],
         "FaceArray1Offset"->CZEncodeTarget[faces[[k]],0.5][[1]],
         "FaceArray2Offset"->CZEncodeTarget[faces[[k]],0.5][[2]],
         "FaceArray3Offset"->CZEncodeTarget[faces[[k]],0.5][[3]]
         ],
      {k,1,Length[faces]}]];


{ trainingSet, validationSet } = { ds[[rnds[[1;;80000]] ]], ds[[rnds[[80001;;]] ]] };


trunk = NetChain[{
   ConvolutionLayer[16,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]
}];
(* trunk has receptive field of size 94x94 *)


block2 = NetChain[{
   PaddingLayer[{{0,0},{0,1},{0,1}}], ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2] } ];
(* block2 has receptive field of size 190x190 *)


block3 = NetChain[{
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp } ];
(* block3 has receptive field of size 382x382 *)


multibox1 = NetChain[ { ConvolutionLayer[1,{1,1}], PartLayer[1], LogisticSigmoid } ];
multibox2 = NetChain[ { ConvolutionLayer[1,{1,1}], PartLayer[1], LogisticSigmoid } ];
multibox3 = NetChain[{ConvolutionLayer[1,{1,1}],PartLayer[1],LogisticSigmoid}];
multibox1Offset = NetChain[ { ConvolutionLayer[1,{1,1}], PartLayer[1], LogisticSigmoid } ];
multibox2Offset = NetChain[ { ConvolutionLayer[1,{1,1}], PartLayer[1], LogisticSigmoid } ];
multibox3Offset = NetChain[{ConvolutionLayer[1,{1,1}],PartLayer[1],LogisticSigmoid}];


net = NetGraph[
   <|"trunk"->trunk,"block2"->block2,"block3"->block3,"multibox1"->multibox1,"multibox2"->multibox2,"multibox3"->multibox3,
   "multibox1Offset"->multibox1Offset,"multibox2Offset"->multibox2Offset,"multibox3Offset"->multibox3Offset|>,
   {"trunk"->"block2"->"block3"->"multibox3"->NetPort["FaceArray3"],"trunk"->"multibox1"->NetPort["FaceArray1"],"block2"->"multibox2"->NetPort["FaceArray2"],
   "trunk"->"multibox1Offset","block2"->"multibox2Offset","block3"->"multibox3Offset",
   "multibox1Offset"->NetPort["FaceArray1Offset"],"multibox2Offset"->NetPort["FaceArray2Offset"],"multibox3Offset"->NetPort["FaceArray3Offset"]},
   "Input"->NetEncoder[{"Image",{640,480},"ColorSpace"->"RGB"}]];


lossNet = NetGraph[ <|
   "net"->net, 
   "L1"->CrossEntropyLossLayer["Binary"], "L2"->CrossEntropyLossLayer["Binary"], "L3"->CrossEntropyLossLayer["Binary"],
   "L1O"->CrossEntropyLossLayer["Binary"], "L2O"->CrossEntropyLossLayer["Binary"], "L3O"->CrossEntropyLossLayer["Binary"] |>, {
   NetPort[{"net","FaceArray1"}] -> NetPort[{"L1","Input"}], NetPort["FaceArray1"]->NetPort[{"L1","Target"}],
   NetPort[{"net","FaceArray2"}] -> NetPort[{"L2","Input"}], NetPort["FaceArray2"]->NetPort[{"L2","Target"}],
   NetPort[{"net","FaceArray3"}] -> NetPort[{"L3","Input"}], NetPort["FaceArray3"]->NetPort[{"L3","Target"}],
   NetPort[{"net","FaceArray1Offset"}] -> NetPort[{"L1O","Input"}], NetPort["FaceArray1Offset"]->NetPort[{"L1O","Target"}],
   NetPort[{"net","FaceArray2Offset"}] -> NetPort[{"L2O","Input"}], NetPort["FaceArray2Offset"]->NetPort[{"L2O","Target"}],
   NetPort[{"net","FaceArray3Offset"}] -> NetPort[{"L3O","Input"}], NetPort["FaceArray3Offset"]->NetPort[{"L3O","Target"}]
    } ];


inet = NetInitialize[lossNet, Method->"Orthogonal"];


trained = NetTrain[ inet, trainingSet, All,
            ValidationSet->validationSet,TargetDevice->"GPU",
            TrainingProgressCheckpointing->{"Directory","c:\\users\\julian\\checkpoint5"}];


(*
   validation .0244, .0237 (2nd round). Note importance of initialisation method.
   Using test:
      Table[CZHighlightFaces[Import@files[[rnds[[k]]]],Threshold\[Rule].5,OverlappingWindows\[Rule]True],{k,80001,80100}]
   achieves 0 false positives and 3 false negatives
*)


Export["c:\\Users\\julian\\TmpFaceDetection.mx",trained];


(*
trained=NetReplacePart[NetExtract[Import["c:\\Users\\julian\\checkpoint5\\2018-05-23T09-02-29_0_02_053334_2.29e-2_2.37e-2.wlnet"],"net"],"Input"\[Rule]NetEncoder[{"Image",{640,480},ColorSpace\[Rule]"RGB"}]];
CloudPut[trained,"VisiNetv1.wlnet"]
Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\VisiNetv1.wlnet",trained]
*)
