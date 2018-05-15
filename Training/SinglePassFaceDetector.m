(* ::Package:: *)

(*
   Training file for a single pass face detector.
   Input image should be of size VGA.
   
   Please note, have had some problems regarding training bug, please ensure that no other kernel
   instances are running, and may wish to try CPU. Not sure if it is a GPU/CPU issue, or problems
   with decoding as we are doing out of core training.
*)


(*
   H.-W. Ng, S. Winkler.
   A data-driven approach to cleaning large face datasets.
   Proc. IEEE International Conference on Image Processing (ICIP), Paris, France, Oct. 27-30, 2014.

   http://vintage.winklerbros.net/facescrub.html
*)


size[face_] := (face[[2,1]]-face[[1,1]])


CZEncoder[faces_] := ReplacePart[
   {ConstantArray[0,{15,20}],ConstantArray[0,{8,10}]},
   Map[ If[size[#] < 190, {1,  15+1-Ceiling[(#[[1,2]]+#[[2,2]])/(2*32)], Ceiling[(#[[1,1]]+#[[2,1]])/(2*32)] },
      {2,  8+1-Ceiling[(#[[1,2]]+#[[2,2]])/(4*32)], Ceiling[(#[[1,1]]+#[[2,1]])/(4*32)] } ]
->1&,faces]];


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


ds = Dataset[RandomSample@Table[
   Association["Input"->files[[k]],
      "FaceArray1"->CZEncoder[faces[[k]]][[1]],
      "FaceArray2"->CZEncoder[faces[[k]]][[2]]],
{k,1,Length[faces]}]];


trunk = NetChain[{
   ConvolutionLayer[16,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]
}];


block2 = NetChain[{
   PaddingLayer[{{0,0},{0,1},{0,1}}], ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2] } ];


multibox1 = NetChain[ { ConvolutionLayer[1,{1,1}], PartLayer[1], LogisticSigmoid } ];
multibox2 = NetChain[ { ConvolutionLayer[1,{1,1}], PartLayer[1], LogisticSigmoid } ];


net = NetGraph[ {
   trunk, multibox1, block2, multibox2 },
   {1->2->NetPort["FaceArray1"], 1->3->4->NetPort["FaceArray2"] },
   "Input"->NetEncoder[{"Image",{640,480},"ColorSpace"->"RGB"}]
];


trained=NetTrain[net,ds[[1;;80000]],All,ValidationSet->ds[[80001;;-1]],TargetDevice->"GPU",TrainingProgressCheckpointing->{"Directory","c:\\users\\julian\\checkpoints"}];


(* .0079 round 1, .00751, round 7 .00728 *)


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\SinglePassTmp.mx",trained];


sz[1]:=57/(32*2.)


sz[2]:=114/(32*2)


(* Takes a net output and returns rectangles *)
CZDecoder[output_]:=Rectangle@@@Map[
   {{32*(#[[3]]-.5)-32*sz[#[[1]]],32*(15-#[[2]]+.5)-32*sz[#[[1]]]},{32*(#[[3]]-.5)+32*sz[#[[1]]],32*(15-#[[2]]+.5)+32*sz[#[[1]]]}}&,
   Position[output,x_/;x>.5]]


CZHighlightFaces[ img_Image ] := HighlightImage[ ConformImages[{img},{640,480},"Fit"][[1]], CZDecoder@trained@(ConformImages[{img},{640,480},"Fit"][[1]]) ]
