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
   "joint2"->{ConvolutionLayer[20,{1,1}],TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]}

},{
   "l1"->"l2"->"l3"->"l4"->"l5"->"joint5"->NetPort["joint5"],
   "l4"->"joint4"->NetPort["joint4"],
   "l3"->"joint3"->NetPort["joint3"],
   "l2"->"joint2"->NetPort["joint2"]
   
},
   "Input"->NetEncoder[{"Image",{512,512},"ColorSpace"->"RGB"}]];


encode[rects_]:=Module[{b2,b3,b4,b5,s2,s3,s4,s5,joint5,joint4,joint3,joint2},
   b2 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]>155&], { 512, 512 }, { 8, 8 }, 64, 0 ];
   b3 = Total[Partition[b2,{2,2}],{-2,-1}];
   b4 = Total[Partition[b3,{2,2}],{-2,-1}];
   b5 = Total[Partition[b4,{2,2}],{-2,-1}];

   s2 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]<=155&], { 512, 512 }, { 8, 8 }, 64, 0 ];
   s3 = Total[Partition[s2,{2,2}],{-2,-1}];
   s4 = Total[Partition[s3,{2,2}],{-2,-1}];
   s5 = Total[Partition[s4,{2,2}],{-2,-1}];

   joint5 = b5*5+s5;
   joint4 = b4*5+s4;
   joint3 = b3*5+s3;
   joint2 = b2*5+s2;
   
   Association[
   "joint5"->joint5+1,
   "joint4"->joint4+1,
   "joint3"->joint3+1,
   "joint2"->joint2+1 ]
];


(* ::Input:: *)
(*vect[n_,len_]:=ReplacePart[ConstantArray[0,len],(n+1)->1]*)


vecta[arr_,len_]:=Map[vect[#,len]&,arr,{2}];


(* ::Input:: *)
(*mf1=CZImageConformer[{512,512},"Fit"]/@Import["~/ImageDataSets/FaceScrub/ActorImages/Original/ActorImages1/*.jpg"];*)
(*mf2=CZImageConformer[{512,512},"Fit"]/@Import["~/ImageDataSets/FaceScrub/ActorImages/Original/ActorImages2/*.jpg"];*)
(*mf=Join[mf1,mf2];*)


r1=FindFaces/@mf;


(* ::Input:: *)
(*dataset=Table[Append[encode[r1[[k]]],"Input"->mf[[k]]],{k,1,Length[mf]}];*)


(* ::Input:: *)
(*SeedRandom[1234];rnds=RandomSample[dataset];*)


Length[dataset]


(* ::Input:: *)
(*{trainingSet,validationSet}={rnds[[;;15000]],rnds[[15001;;]]};*)


Length[mf1]


n1


decoder[assoc_]:=First[Ordering[assoc["total"][[All,1,1]],-1]-1]


(* ::Input:: *)
(*a=First[MaximalBy[Flatten[Table[*)
(*{b5,m5,s5,res["big5"][[b5+1,1,1]]*res["medium5"][[m5+1,1,1]]*res["small5"][[s5+1,1,1]]},{b5,0,2},{m5,0,2-b5},{s5,2-(b5+m5),2-(b5+m5)}],2],Last]][[1;;3]]*)


(* ::Input:: *)
(*(**)
(*   matrix is going to be 2*2*count*)
(*   total is the total to be partition'd up.*)
(**)*)
(*allocate[matrix_,tot_]:=*)
(*Partition[First[MaximalBy[Flatten[Table[{n11,n12,n21,n22,matrix[[1,1,n11+1]]*matrix[[1,2,n12+1]]*matrix[[2,1,n21+1]]*matrix[[2,2,n22+1]]},{n11,0,tot},{n12,0,tot-n11},{n21,0,tot-(n11+n12)},{n22,tot-(n11+n12+n21),tot-(n11+n12+n21)}],3],Last]][[1;;4]],2]*)


(* ::Input:: *)
(*unpartition[matrices_]:=ImageData[ImageAssemble[Map[Image,matrices,{2}]]]*)


(* ::Input:: *)
(*allocate[Transpose[res["medium4"],{3,1,2}],1]*)


(* ::Input:: *)
(*mp3=Partition[Transpose[res["medium3"],{3,1,2}],{2,2}];mp3//Dimensions*)


(* ::Input:: *)
(*mr3=unpartition[MapThread[allocate,{mp3,allocate[Transpose[res["medium4"],{3,1,2}],1]},2]];mr3//MatrixForm*)


(* ::Input:: *)
(*mp2=Partition[Transpose[res["medium2"],{3,1,2}],{2,2}];mp2//Dimensions*)


(* ::Input:: *)
(*mr2=unpartition[MapThread[allocate,{mp2,mr3},2]];mr2//MatrixForm*)


(* ::Input:: *)
(*conv[c_]:={Rectangle[{c[[2]],512-c[[1]]}-{50,50},{c[[2]],512-c[[1]]}+{50,50}]}*)


(* ::Input:: *)
(*decoder[assoc_]:=( *)
(*n5=First[Ordering[Total[Partition[assoc["joint5"][[1,1]],5],{2}],-1]]-1;*)
(*s5=First[Ordering[Partition[assoc["joint5"][[1,1]],5][[n5+1]],-1]]-1;*)
(*o1=If[n5==1,{Rectangle[{256,256}-{200,200}/2,{256,256}+{200,200}/2]},{}];*)
(*o2=Map[conv[(#-{.5,.5})*256]&,Position[allocate[Map[Partition[#,5]&,assoc["joint4"],{2}][[All,All,1]],s5],1]];*)
(*Join[o1,o2])*)


(* ::Input:: *)
(*large[jointarray_]:=Map[Total[Partition[#,5],{2}]&,jointarray,{2}]*)


(* ::Input:: *)
(*classify[vector_]:=First[Ordering[vector,-1]]-1*)


decoder[assoc_]:=(
   l5=Map[classify,large[assoc["joint5"]][[All,All]],{2}];
   l4=unpartition[MapThread[allocate,{Partition[large[assoc["joint4"]],{2,2}],l5},2]];
   l3=unpartition[MapThread[allocate,{Partition[large[assoc["joint3"]],{2,2}],l4},2]];
   l2=unpartition[MapThread[allocate,{Partition[large[assoc["joint2"]],{2,2}],l3},2]];
   s5=Map[classify,small[assoc["joint5"],l5],{2}];
   s4=unpartition[MapThread[allocate,{Partition[small[assoc["joint4"],l4],{2,2}],s5},2]];
   s3=unpartition[MapThread[allocate,{Partition[small[assoc["joint3"],l3],{2,2}],s4},2]];
   s2=unpartition[MapThread[allocate,{Partition[small[assoc["joint2"],l2],{2,2}],s3},2]];
   <|"Big"->l2,"Small"->s2|>
   );


render[ assoc_ ] := Join[
   Map[
         Rectangle[{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}+{100,100}]&,
      Position[assoc["Big"],x_/;x>0]],
   Map[
         Rectangle[{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}-{50,50},{64*(#[[2]]-.5),512-64*(#[[1]]-.5)}+{50,50}]&,
      Position[assoc["Small"],x_/;x>0]]
      ];      


totdec[assoc_]:=render[ decoder[ assoc ] ]


decoder[oi]


l3//Dimensions


l4==l4test


Export["~/Google Drive/Personal/Computer Science/CZModels/CountNet2.wlnet",trained]



