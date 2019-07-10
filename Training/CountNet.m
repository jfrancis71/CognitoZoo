(* ::Package:: *)

(* Experimental in development, not ready for use *)
(* Just here so it can be change management tracked *)

<<CZUtils.m


trunk = NetChain[{
   ConvolutionLayer[16,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[32,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[64,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[128,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2],
   ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]
}];
(* trunk has receptive field of size 94x94 *)


n1 = NetGraph[{
   "l1"->trunk,
   "l2"->{ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]},
   "l3"->{ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]},
   "l4"->{ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]},
   "l5"->{ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]},
   "joint5"->{ConvolutionLayer[20,{1,1}],TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]},
   "joint4"->{ConvolutionLayer[20,{1,1}],TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]},
   "joint3"->{ConvolutionLayer[20,{1,1}],TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]},
   "joint2"->{ConvolutionLayer[20,{1,1}],TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]},
   "joint1"->{ConvolutionLayer[20,{1,1}],TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]}

},{
   "l1"->"l2"->"l3"->"l4"->"l5"->"joint5"->NetPort["joint5"],
   "l4"->"joint4"->NetPort["joint4"],
   "l3"->"joint3"->NetPort["joint3"],
   "l2"->"joint2"->NetPort["joint2"],
   "l1"->"joint1"->NetPort["joint1"]
},
   "Input"->NetEncoder[{"Image",{512,512},"ColorSpace"->"RGB"}]];


size[ face_ ] := (face[[2,1]]-face[[1,1]])


CZReplacePart[array_,rules_] := ReplacePart[ array, Select[ rules, #[[1,1]] > 0 && #[[1,2]] > 0 & ] ]


CZCentroidsToArray[ centroids_, inputDims_, arrayDims_, stride_, offset_ ] :=
   CZReplacePart[ ConstantArray[ 0, arrayDims ], kt=Map[{Ceiling[(1+inputDims[[2]]-#[[2]])/stride+offset],(1+Floor[(#[[1]]-1)/stride-offset])}->1&,centroids] ]


encode[rects_]:=Module[{b1,b2,b3,b4,b5,s1,s2,s3,s4,s5,joint5,joint4,joint3,joint2,joint1},
   b1 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]>155&], { 512, 512 }, { 16, 16 }, 32, 0 ];
   b2 = Total[Partition[b1,{2,2}],{-2,-1}];
   b3 = Total[Partition[b2,{2,2}],{-2,-1}];
   b4 = Total[Partition[b3,{2,2}],{-2,-1}];
   b5 = Total[Partition[b4,{2,2}],{-2,-1}];

   s1 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]<=155&], { 512, 512 }, { 16, 16 }, 32, 0 ];
   s2 = Total[Partition[s1,{2,2}],{-2,-1}];
   s3 = Total[Partition[s2,{2,2}],{-2,-1}];
   s4 = Total[Partition[s3,{2,2}],{-2,-1}];
   s5 = Total[Partition[s4,{2,2}],{-2,-1}];

   joint5 = b5*5+s5;
   joint4 = b4*5+s4;
   joint3 = b3*5+s3;
   joint2 = b2*5+s2;
   joint1 = b1*5+s1;
   
   Association[
   "joint5"->joint5+1,
   "joint4"->joint4+1,
   "joint3"->joint3+1,
   "joint2"->joint2+1,
   "joint1"->joint1+1 ]
];


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


faces1=Import["~/ImageDataSets/FaceScrub/ActorImages/MMFaceDetectActorImages1.mx"];
faces2=Import["~/ImageDataSets/FaceScrub/ActorImages/MMFaceDetectActorImages2.mx"];
faces3=Import["~/ImageDataSets/FaceScrub/ActorImages/MMFaceDetectActorImages3.mx"];
faces4=Import["~/ImageDataSets/FaceScrub/ActorImages/MMFaceDetectActorImages4.mx"];
faces5=Import["~/ImageDataSets/FaceScrub/ActorImages/MMFaceDetectActorImages5.mx"];
faces6=Import["~/ImageDataSets/FaceScrub/ActorImages/MMFaceDetectActorImages6.mx"];
faces7=Import["~/ImageDataSets/FaceScrub/ActressImages/MMFaceDetectActressImages1.mx"];
faces8=Import["~/ImageDataSets/FaceScrub/ActressImages/MMFaceDetectActressImages2.mx"];
faces9=Import["~/ImageDataSets/FaceScrub/ActressImages/MMFaceDetectActressImages3.mx"];
faces10=Import["~/ImageDataSets/FaceScrub/ActressImages/MMFaceDetectActressImages4.mx"];
faces11=Import["~/ImageDataSets/FaceScrub/ActressImages/MMFaceDetectActressImages5.mx"];
faces12=Import["~/ImageDataSets/FaceScrub/ActressImages/MMFaceDetectActressImages6.mx"];

faces=Join[faces1,faces2,faces3,faces4,faces5,faces6,faces7,faces8,faces9,faces10,faces11,faces12];


dataset=Table[Append[encode[faces[[k]]],"Input"->File[files[[k]]]],{k,1,Length[files]}];


SeedRandom[1234];rnds=RandomSample[dataset];


Length[dataset]


{trainingSet,validationSet}={rnds[[;;84000]],rnds[[84001;;]]};


(*
   matrix is going to be 2*2*count
   total is the total to be partition'd up.
*)
allocate[matrix_,tot_]:=
Partition[First[MaximalBy[Flatten[Table[{n11,n12,n21,n22,matrix[[1,1,n11+1]]*matrix[[1,2,n12+1]]*matrix[[2,1,n21+1]]*matrix[[2,2,n22+1]]},{n11,0,tot},{n12,0,tot-n11},{n21,0,tot-(n11+n12)},{n22,tot-(n11+n12+n21),tot-(n11+n12+n21)}],3],Last]][[1;;4]],2]


unpartition[matrices_]:=ImageData[ImageAssemble[Map[Image,matrices,{2}]]]


large[jointarray_]:=Map[Total[Partition[#,5],{2}]&,jointarray,{2}]


small[jointarray_,large_]:=MapThread[Partition[#1,5][[#2+1]]&,{jointarray,large},2];


classify[vector_]:=First[Ordering[vector,-1]]-1


decoder[assoc_]:=(
   l5=Map[classify,large[assoc["joint5"]][[All,All]],{2}];
   l4=unpartition[MapThread[allocate,{Partition[large[assoc["joint4"]],{2,2}],l5},2]];
   l3=unpartition[MapThread[allocate,{Partition[large[assoc["joint3"]],{2,2}],l4},2]];
   l2=unpartition[MapThread[allocate,{Partition[large[assoc["joint2"]],{2,2}],l3},2]];
   l1=unpartition[MapThread[allocate,{Partition[large[assoc["joint1"]],{2,2}],l2},2]];
   s5=Map[classify,small[assoc["joint5"],l5],{2}];
   s4=unpartition[MapThread[allocate,{Partition[small[assoc["joint4"],l4],{2,2}],s5},2]];
   s3=unpartition[MapThread[allocate,{Partition[small[assoc["joint3"],l3],{2,2}],s4},2]];
   s2=unpartition[MapThread[allocate,{Partition[small[assoc["joint2"],l2],{2,2}],s3},2]];
   s1=unpartition[MapThread[allocate,{Partition[small[assoc["joint1"],l1],{2,2}],s2},2]];
   <|"Big"->l1,"Small"->s1|>
   );


render[ assoc_ ] := Join[
   Map[
         Rectangle[{32*(#[[2]]-.5),512-32*(#[[1]]-.5)}-{100,100},{32*(#[[2]]-.5),512-32*(#[[1]]-.5)}+{100,100}]&,
      Position[assoc["Big"],x_/;x>0]],
   Map[
         Rectangle[{32*(#[[2]]-.5),512-32*(#[[1]]-.5)}-{50,50},{32*(#[[2]]-.5),512-32*(#[[1]]-.5)}+{50,50}]&,
      Position[assoc["Small"],x_/;x>0]]
      ];      


totdec[assoc_]:=render[ decoder[ assoc ] ]


(* Export["~/Google Drive/Personal/Computer Science/CZModels/CountNet2.wlnet",trained]*)
