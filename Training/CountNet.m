(* ::Package:: *)

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
   "l5"->{ConvolutionLayer[256,{2,2}],Ramp},
   "small1"->{ConvolutionLayer[2,{1,1}],SoftmaxLayer[1]},
   "small2"->{ConvolutionLayer[5,{1,1}],SoftmaxLayer[1]},
   "small3"->{ConvolutionLayer[17,{1,1}],SoftmaxLayer[1]},
   "small4"->{ConvolutionLayer[65,{1,1}],SoftmaxLayer[1]},
   "small5"->{ConvolutionLayer[257,{1,1}],SoftmaxLayer[1]},
   "medium2"->{ConvolutionLayer[2,{1,1}],SoftmaxLayer[1]},
   "big2"->{ConvolutionLayer[2,{1,1}],SoftmaxLayer[1]},
   "medium3"->{ConvolutionLayer[5,{1,1}],SoftmaxLayer[1]},
   "big3"->{ConvolutionLayer[5,{1,1}],SoftmaxLayer[1]},
   "medium4"->{ConvolutionLayer[17,{1,1}],SoftmaxLayer[1]},
   "big4"->{ConvolutionLayer[17,{1,1}],SoftmaxLayer[1]},
   "medium5"->{ConvolutionLayer[65,{1,1}],SoftmaxLayer[1]},
   "big5"->{ConvolutionLayer[65,{1,1}],SoftmaxLayer[1]},
   "total"->{ConvolutionLayer[64+64+256+1,{1,1}],SoftmaxLayer[1]}
},{
   "l1"->"l2"->"l3"->"l4"->"l5"->"total",
   "l1"->"small1",
   "l2"->{"small2","medium2","big2"},
   "l3"->{"small3","medium3","big3"},
   "l4"->{"small4","medium4","big4"},
   "l5"->{"small5","medium5","big5"},
   "small1"->NetPort["small1"],
   "small2"->NetPort["small2"],
   "small3"->NetPort["small3"],
   "small4"->NetPort["small4"],
   "small5"->NetPort["small5"],
   "medium2"->NetPort["medium2"],
   "medium3"->NetPort["medium3"],
   "medium4"->NetPort["medium4"],
   "medium5"->NetPort["medium5"],
   "big2"->NetPort["big2"],
   "big3"->NetPort["big3"],
   "big4"->NetPort["big4"],
   "big5"->NetPort["big5"],
   "total"->NetPort["total"]
},
   "Input"->NetEncoder[{"Image",{512,512},"ColorSpace"->"RGB"}]];


lossnet=NetGraph[
<|"net"->n1,
"smallt1"->TransposeLayer[{3<->1,1<->2}],"smallt2"->TransposeLayer[{3<->1,1<->2}],"smallt3"->TransposeLayer[{3<->1,1<->2}],"smallt4"->TransposeLayer[{3<->1,1<->2}],"smallt5"->TransposeLayer[{3<->1,1<->2}],
"smallloss1"->CrossEntropyLossLayer["Index"],"smallloss2"->CrossEntropyLossLayer["Index"],"smallloss3"->CrossEntropyLossLayer["Index"],"smallloss4"->CrossEntropyLossLayer["Index"],"smallloss5"->CrossEntropyLossLayer["Index"],
"mediumt2"->TransposeLayer[{3<->1,1<->2}],"mediumt3"->TransposeLayer[{3<->1,1<->2}],"mediumt4"->TransposeLayer[{3<->1,1<->2}],"mediumt5"->TransposeLayer[{3<->1,1<->2}],
"mediumloss2"->CrossEntropyLossLayer["Index"],"mediumloss3"->CrossEntropyLossLayer["Index"],"mediumloss4"->CrossEntropyLossLayer["Index"],"mediumloss5"->CrossEntropyLossLayer["Index"],
"bigt2"->TransposeLayer[{3<->1,1<->2}],"bigt3"->TransposeLayer[{3<->1,1<->2}],"bigt4"->TransposeLayer[{3<->1,1<->2}],"bigt5"->TransposeLayer[{3<->1,1<->2}],
"bigloss2"->CrossEntropyLossLayer["Index"],"bigloss3"->CrossEntropyLossLayer["Index"],"bigloss4"->CrossEntropyLossLayer["Index"],"bigloss5"->CrossEntropyLossLayer["Index"],
"totallosst"->TransposeLayer[{3<->1,1<->2}],
"totalloss"->CrossEntropyLossLayer["Index"]
|>,
{NetPort["Input"]->"net",

NetPort[{"net","small1"}]->"smallt1"->NetPort[{"smallloss1","Input"}],
NetPort["small1"]->NetPort[{"smallloss1","Target"}],
NetPort[{"net","small2"}]->"smallt2"->NetPort[{"smallloss2","Input"}],
NetPort["small2"]->NetPort[{"smallloss2","Target"}],
NetPort[{"net","small3"}]->"smallt3"->NetPort[{"smallloss3","Input"}],
NetPort["small3"]->NetPort[{"smallloss3","Target"}],
NetPort[{"net","small4"}]->"smallt4"->NetPort[{"smallloss4","Input"}],
NetPort["small4"]->NetPort[{"smallloss4","Target"}],
NetPort[{"net","small5"}]->"smallt5"->NetPort[{"smallloss5","Input"}],
NetPort["small5"]->NetPort[{"smallloss5","Target"}],

NetPort[{"net","medium2"}]->"mediumt2"->NetPort[{"mediumloss2","Input"}],
NetPort["medium2"]->NetPort[{"mediumloss2","Target"}],
NetPort[{"net","medium3"}]->"mediumt3"->NetPort[{"mediumloss3","Input"}],
NetPort["medium3"]->NetPort[{"mediumloss3","Target"}],
NetPort[{"net","medium4"}]->"mediumt4"->NetPort[{"mediumloss4","Input"}],
NetPort["medium4"]->NetPort[{"mediumloss4","Target"}],
NetPort[{"net","medium5"}]->"mediumt5"->NetPort[{"mediumloss5","Input"}],
NetPort["medium5"]->NetPort[{"mediumloss5","Target"}],

NetPort[{"net","big2"}]->"bigt2"->NetPort[{"bigloss2","Input"}],
NetPort["big2"]->NetPort[{"bigloss2","Target"}],
NetPort[{"net","big3"}]->"bigt3"->NetPort[{"bigloss3","Input"}],
NetPort["big3"]->NetPort[{"bigloss3","Target"}],
NetPort[{"net","big4"}]->"bigt4"->NetPort[{"bigloss4","Input"}],
NetPort["big4"]->NetPort[{"bigloss4","Target"}],
NetPort[{"net","big5"}]->"bigt5"->NetPort[{"bigloss5","Input"}],
NetPort["big5"]->NetPort[{"bigloss5","Target"}],

NetPort[{"net","total"}]->"totallosst"->NetPort[{"totalloss","Input"}],
NetPort["total"]->NetPort[{"totalloss","Target"}]

}]


soft[array_,range_]:=Transpose[Map[ReplacePart[ConstantArray[0,range],(#+1)->1]&,array,{2}],{2,3,1}]


encode[rects_]:=Module[{sm1,sm2,sm3,sm4,sm5,m2,m3,m4,m5,b2,b3,b4,b5,total},
   sm1 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]<108&], { 512, 512 }, { 16, 16 }, 32, 0 ];
   sm2 = Total[Partition[sm1,{2,2}],{-2,-1}];
   sm3 = Total[Partition[sm2,{2,2}],{-2,-1}];
   sm4 = Total[Partition[sm3,{2,2}],{-2,-1}];
   sm5 = Total[Partition[sm4,{2,2}],{-2,-1}];
   m2 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]>=108&&size[#]<=155&], { 512, 512 }, { 8, 8 }, 64, 0 ];
   m3 = Total[Partition[m2,{2,2}],{-2,-1}];
   m4 = Total[Partition[m3,{2,2}],{-2,-1}];
   m5 = Total[Partition[m4,{2,2}],{-2,-1}];
   
   b2 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]>155&], { 512, 512 }, { 8, 8 }, 64, 0 ];
   b3 = Total[Partition[b2,{2,2}],{-2,-1}];
   b4 = Total[Partition[b3,{2,2}],{-2,-1}];
   b5 = Total[Partition[b4,{2,2}],{-2,-1}];
   total = sm5+m5+b5;
   
   Association[
   "small1"->sm1+1,"small2"->sm2+1,"small3"->sm3+1,"small4"->sm4+1,"small5"->sm5+1,
   "medium2"->m2+1,"medium3"->m3+1,"medium4"->m4+1,"medium5"->m5+1,
   "big2"->b2+1,"big3"->b3+1,"big4"->b4+1,"big5"->b5+1,
   "total"->total+1
   ]
];


(* ::Input:: *)
(*mf1=CZImageConformer[{512,512},"Fit"]/@Import["~/ImageDataSets/FaceScrub/ActorImages/Original/ActorImages1/*.jpg"];*)


r1=FindFaces/@mf1;


(* ::Input:: *)
(*dataset=Table[Association["Input"->mf1[[k]],*)
(*"small1"->encode[r1[[k]]]["small1"],*)
(*"small2"->encode[r1[[k]]]["small2"],*)
(*"small3"->encode[r1[[k]]]["small3"],*)
(*"small4"->encode[r1[[k]]]["small4"],*)
(*"small5"->encode[r1[[k]]]["small5"],*)
(**)
(*"medium2"->encode[r1[[k]]]["medium2"],*)
(*"medium3"->encode[r1[[k]]]["medium3"],*)
(*"medium4"->encode[r1[[k]]]["medium4"],*)
(*"medium5"->encode[r1[[k]]]["medium5"],*)
(**)
(**)
(*"big2"->encode[r1[[k]]]["big2"],*)
(*"big3"->encode[r1[[k]]]["big3"],*)
(*"big4"->encode[r1[[k]]]["big4"],*)
(*"big5"->encode[r1[[k]]]["big5"],*)
(*"total"->encode[r1[[k]]]["total"]],{k,1,Length[mf1]}];*)


(* ::Input:: *)
(*SeedRandom[1234];rnds=RandomSample[dataset];*)


(* ::Input:: *)
(*{trainingSet,validationSet}={rnds[[;;7500]],rnds[[7501;;]]};*)


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
