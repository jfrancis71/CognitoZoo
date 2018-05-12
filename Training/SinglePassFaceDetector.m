(* ::Package:: *)

sz1[face_]:=If[size[face]>190,2,1];


size[face_] := (face[[2,1]]-face[[1,1]])


CZEncoder[faces_]:=ReplacePart[ConstantArray[0,{2,15,20}],Map[{
sz1[#],
15+1-Ceiling[(#[[1,2]]+#[[2,2]])/(2*32)],
Ceiling[(#[[1,1]]+#[[2,1]])/(2*32)]
}->1&,faces]];


files=Map[File,FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\ActorImages1\\*.jpg"]];
faces=Import["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\VGA\\DLibFaces1.mx"];


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


trained=NetTrain[net,ndataset[[1;;7000]],ValidationSet->ndataset[[7001;;-1]],TargetDevice->"GPU"];


(* 7,000
   Significant learning within 1 batch. Achieved loss .00488, error 0.136%
   Stopped after 3 training rounds (still learning)
*)


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\SinglePassTmp.wlnet",trained];


sz[1]:=57/(32*2.)


sz[2]:=114/(32*2)


(* Takes a net output and returns rectangles *)
CZDecoder[output_]:=Rectangle@@@Map[
   {{32*(#[[3]]-.5)-32*sz[#[[1]]],32*(15-#[[2]]+.5)-32*sz[#[[1]]]},{32*(#[[3]]-.5)+32*sz[#[[1]]],32*(15-#[[2]]+.5)+32*sz[#[[1]]]}}&,
   Position[output,x_/;x>.5]]


CZHighlightFaces[ img_Image ] := HighlightImage[ ConformImages[{img},{640,480},"Fit"][[1]], CZDecoder@trained@(ConformImages[{img},{640,480},"Fit"][[1]]) ]
