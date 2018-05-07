(* ::Package:: *)

<<CZUtils.m


<<CZDLibFace.m


(* Note this is better than importing all eg *.jpg and then resizing, speed is similar, but memory balloons.
   Takes around 3 mins to import around 8,000 images *)
ImportFromDir[filePath_String] :=
   Map[
      ImageResize[CZImagePadToSquare[ColorConvert[Import[#],"Grayscale"]],256]&,
      FileNames[filePath]];


(* faces=Map[FindFaces,images]; *)


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
];


trained=NetTrain[net,ndataset[[1;;80000]],ValidationSet->ndataset[[80001;;-1]],TargetDevice->"GPU"];


(*
   Validation scores:   7,000 .011
                        16,000 .0091
                        40, 000 .0075
                        80,000  .0065 (early stopping) NB this is where female data included
*)


Export["c:\\Users\\julian\\Google Drive\\Personal\\Computer Science\\CZModels\\SinglePassTmp.wlnet",trained];


sz[1]:=57/(32*2.)


sz[2]:=114/(32*2)


(* Takes a net output and returns rectangles *)
CZDecoder[output_]:=Rectangle@@@Map[32*{{#[[3]]-.5-sz[#[[1]]],8-#[[2]]+.5-sz[#[[1]]]},{#[[3]]-.5+sz[#[[1]]],8-#[[2]]+.5+sz[#[[1]]]}}&,Position[output,x_/;x>.5]]


CZHighlightFaces[ img_Image ] := Module[{std=ColorConvert[ImageResize[CZImagePadToSquare@img,256],"Grayscale"]},
   HighlightImage[ std, CZDecoder@trained@std ] ]
