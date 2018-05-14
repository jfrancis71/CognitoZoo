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


sz1[face_]:=If[size[face]>190,2,1];


size[face_] := (face[[2,1]]-face[[1,1]])


CZEncoder[faces_]:=ReplacePart[ConstantArray[0,{2,15,20}],Map[{
sz1[#],
15+1-Ceiling[(#[[1,2]]+#[[2,2]])/(2*32)],
Ceiling[(#[[1,1]]+#[[2,1]])/(2*32)]
}->1&,faces]];


files=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages1\\*.jpg"]];
faces=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces1.mx"];


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


files=Join[mfiles1,mfiles2,mfiles3,mfiles4];
faces=Join[mfaces1,mfaces2,mfaces3,mfaces4];


dataset=RandomSample[Table[files[[f]]->faces[[f]],{f,1,Length[faces]}]];


ndataset=Map[#[[1]]->CZEncoder[#[[2]]]&,dataset];


trunk = NetChain[{
   ConvolutionLayer[16,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]
}];


net = NetGraph[
   { trunk, ConvolutionLayer[2,{1,1}], LogisticSigmoid },
   { 1->2, 2->3 },
   "Input"->NetEncoder[{"Image",{640,480},"ColorSpace"->"RGB"}]
];   


trained=NetTrain[net,ndataset[[1;;80000]],All,ValidationSet->ndataset[[80001;;-1]],TargetDevice->"GPU",TrainingProgressCheckpointing->{"Directory","c:\\users\\julian\\checkpoints"}];


(* Achieves around 0.1% error rate. Should learn quite quickly within first round, five rounds is optimal. In practice works quite well *)


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\SinglePassTmp.mx",trained];


sz[1]:=57/(32*2.)


sz[2]:=114/(32*2)


(* Takes a net output and returns rectangles *)
CZDecoder[output_]:=Rectangle@@@Map[
   {{32*(#[[3]]-.5)-32*sz[#[[1]]],32*(15-#[[2]]+.5)-32*sz[#[[1]]]},{32*(#[[3]]-.5)+32*sz[#[[1]]],32*(15-#[[2]]+.5)+32*sz[#[[1]]]}}&,
   Position[output,x_/;x>.5]]


CZHighlightFaces[ img_Image ] := HighlightImage[ ConformImages[{img},{640,480},"Fit"][[1]], CZDecoder@trained@(ConformImages[{img},{640,480},"Fit"][[1]]) ]
