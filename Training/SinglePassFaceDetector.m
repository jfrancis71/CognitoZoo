(* ::Package:: *)

files1=FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\ActorImages1\\*.jpg"];


files2=FileNames["C:\\Users\\julian\\ImageDataSets\\FaceScrub\\ActorImages\\ActorImages2\\*.jpg"];


files3=FileNames["C:\\Users\\julian\\ImageDataSets\\ILSVRC\\Data\\DET\\train\\ILSVRC2013_train\\*\\*.JPEG"];


files=Join[files1,files2,RandomSample[files3][[1;;16000]]];


images=Map[ImageResize[CZImagePadToSquare[ColorConvert[Import[#],"Grayscale"]],256]&,files[[1;;-1]]];


faces=Map[FindFaces,images];


sz1[face_]:=If[size[face]>84,2,1];


size[face_] := (face[[2,1]]-face[[1,1]])


CZEncoder[faces_]:=ReplacePart[ConstantArray[0,{2,8,8}],Map[{
sz1[#],
9-Ceiling[(#[[1,2]]+#[[2,2]])/(2*32)],
Ceiling[(#[[1,1]]+#[[2,1]])/(2*32)]
}->1&,faces]];


dataset=RandomSample[Table[images[[f]]->faces[[f]],{f,1,Length[faces]}]];


ndataset=Map[#[[1]]->CZEncoder[#[[2]]]&,dataset];


net=NetChain[{
   ConvolutionLayer[16,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[2,{1,1}],
   LogisticSigmoid
},
   "Input"->NetEncoder[{"Image",{256,256},"ColorSpace"->"Grayscale"}]
]


trained=NetTrain[net,ndataset[[1;;30000]],ValidationSet->ndataset[[30001;;-1]]];


.00954,.00846,.0074


(*Select[Map[#[[2,1]]-#[[1,1]]&,Flatten[faces,1]],#<84&]//Mean*)


(*Select[Map[#[[2,1]]-#[[1,1]]&,Flatten[faces,1]],#>84&]//Mean*)


sz[1]:=57/(32*2.)


sz[2]:=114/(32*2)


(* Takes a net output and returns rectangles *)
CZDecoder[output_]:=Rectangle@@@Map[32*{{#[[3]]-.5-sz[#[[1]]],8-#[[2]]+.5-sz[#[[1]]]},{#[[3]]-.5+sz[#[[1]]],8-#[[2]]+.5+sz[#[[1]]]}}&,Position[output,x_/;x>.5]]
