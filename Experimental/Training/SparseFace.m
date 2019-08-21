(* ::Package:: *)

(* Experimental in development, not ready for use *)
(* Just here so it can be change management tracked *)

<<CZUtils.m


trunk = NetChain[{
   ConvolutionLayer[16,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{1,1}],LogisticSigmoid
},
   "Input"->NetEncoder[{"Image",{512,512},"ColorSpace"->"RGB"}]];


NetReplacePart[trunk,"Input"->{3,512,512}];


(* Works well not quite right train/valid split but in practice works nicely with validation error .001 *)


size[ face_ ] := (face[[2,1]]-face[[1,1]])


CZReplacePart[array_,rules_] := ReplacePart[ array, Select[ rules, #[[1,1]] > 0 && #[[1,2]] > 0 & ] ]


CZCentroidsToArray[ centroids_, inputDims_, arrayDims_, stride_, offset_ ] :=
   CZReplacePart[ ConstantArray[ 0, arrayDims ], kt=Map[{Ceiling[(1+inputDims[[2]]-#[[2]])/stride+offset],(1+Floor[(#[[1]]-1)/stride-offset])}->1&,centroids] ]


codebook=Import["~/Google Drive/Personal/Computer Science/CZModels/SparseFace/codebook1.mx"];


CZEncodeTarget[ faces_ ]:=(
   s1 = CZCentroidsToArray[ RegionCentroid/@Select[faces,size[#]<108&], { 512, 512 }, { 16, 16 }, 32, 0 ];
   m1 = CZCentroidsToArray[ RegionCentroid/@Select[faces,size[#]>=108&&size[#]<=155&], { 512, 512 }, { 8, 8 }, 64, 0 ];
   l1 = CZCentroidsToArray[ RegionCentroid/@Select[faces,size[#]>155&], { 512, 512 }, { 8, 8 }, 64, 0 ];
   a1 = Map[Total@Prepend[Extract[codebook[[1]],Position[#,1]],ConstantArray[0,32]]&,Partition[s1,{8,8}],{2}];
   a2 = Map[Total@Prepend[Extract[codebook[[2]],Position[#,1]],ConstantArray[0,32]]&,Partition[m1,{4,4}],{2}];
   a3 = Map[Total@Prepend[Extract[codebook[[3]],Position[#,1]],ConstantArray[0,32]]&,Partition[l1,{4,4}],{2}];
   Transpose[UnitStep[a1+a2+a3-.001],{2,3,1}]
);


files1=FileNames["~/ImageDataSets/FaceScrub/ActorImages/512/ActorImages1/*.jpg"];
files2=FileNames["~/ImageDataSets/FaceScrub/ActorImages/512/ActorImages2/*.jpg"];
files3=FileNames["~/ImageDataSets/FaceScrub/ActorImages/512/ActorImages3/*.jpg"];
files4=FileNames["~/ImageDataSets/FaceScrub/ActorImages/512/ActorImages4/*.jpg"];
files5=FileNames["~/ImageDataSets/FaceScrub/ActorImages/512/ActorImages5/*.jpg"];
files6=FileNames["~/ImageDataSets/FaceScrub/ActorImages/512/ActorImages6/*.jpg"];
files7=FileNames["~/ImageDataSets/FaceScrub/ActressImages/512/ActressImages1/*.jpg"];
files8=FileNames["~/ImageDataSets/FaceScrub/ActressImages/512/ActressImages2/*.jpg"];
files9=FileNames["~/ImageDataSets/FaceScrub/ActressImages/512/ActressImages3/*.jpg"];
files10=FileNames["~/ImageDataSets/FaceScrub/ActressImages/512/ActressImages4/*.jpg"];
files11=FileNames["~/ImageDataSets/FaceScrub/ActressImages/512/ActressImages5/*.jpg"];
files12=FileNames["~/ImageDataSets/FaceScrub/ActressImages/512/ActressImages6/*.jpg"];
files=Join[files1,files2,files3,files4,files5,files6,files7,files8,files9,files10,files11,files12];


faces1=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibFaceDetectActor1.mx"];
faces2=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibFaceDetectActor2.mx"];
faces3=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibFaceDetectActor3.mx"];
faces4=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibFaceDetectActor4.mx"];
faces5=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibFaceDetectActor5.mx"];
faces6=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibFaceDetectActor6.mx"];
faces7=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibFaceDetectActress1.mx"];
faces8=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibFaceDetectActress2.mx"];
faces9=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibFaceDetectActress3.mx"];
faces10=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibFaceDetectActress4.mx"];
faces11=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibFaceDetectActress5.mx"];
faces12=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibFaceDetectActress6.mx"];
faces=Join[faces1,faces2,faces3,faces4,faces5,faces6,faces7,faces8,faces9,faces10,faces11,faces12];


dataset = Table[File[files[[k]]]->CZEncodeTarget[faces[[k]]],{k,1,Length[files]}];


rnds = Import["~/ImageDataSets/FaceScrub/TrainingRandomisation.mx"];


{ trainingSet, validationSet } = { dataset[[rnds[[1;;80000]] ]], dataset[[rnds[[80001;;]] ]] };


(* Export["~/Google Drive/Personal/Computer Science/CZModels/CountNet2.wlnet",trained]*)


(*
trained = NetTrain[
   trunk,
   trainingSet,
   ValidationSet->validationSet,
   TrainingProgressCheckpointing->{"Directory","~/Google Drive/Personal/Computer Science/CZModels/SparseFace/"},
   TrainingProgressReporting\[Rule]{File["~/Google Drive/Personal/Computer Science/CZModels/SparseFace/results.csv"],"Interval"\[Rule]Quantity[20,"Minutes"]}];
*)
