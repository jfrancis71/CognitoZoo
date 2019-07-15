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


decode1 = NetGraph[{
"j"->CatenateLayer[],
   "c1"->ConvolutionLayer[16,{1,1}],"r1"->Ramp,
   "c2"->ConvolutionLayer[1,{1,1}],"f"->PartLayer[1],"o"->LogisticSigmoid
},{
   "j"->"c1"->"r1"->"c2"->"f"->"o",
   {NetPort["internal"],NetPort["conditional"]}->"j"
},"internal"->{256,15,20},"conditional"->{9,15,20}];


decode2 = NetGraph[{
"j"->CatenateLayer[],
   "c1"->ConvolutionLayer[16,{1,1}],"r1"->Ramp,
   "c2"->ConvolutionLayer[1,{1,1}],"f"->PartLayer[1],"o"->LogisticSigmoid
},{
   "j"->"c1"->"r1"->"c2"->"f"->"o",
   {NetPort["internal"],NetPort["conditional"]}->"j"
},"internal"->{256,8,10},"conditional"->{13,8,10}];


decode3 = NetGraph[{
"j"->CatenateLayer[],
   "c1"->ConvolutionLayer[16,{1,1}],"r1"->Ramp,
   "c2"->ConvolutionLayer[1,{1,1}],"f"->PartLayer[1],"o"->LogisticSigmoid
},{
   "j"->"c1"->"r1"->"c2"->"f"->"o",
   {NetPort["internal"],NetPort["conditional"]}->"j"
},"internal"->{256,8,10},"conditional"->{9,8,10}];


n1 = NetGraph[{
   "l1"->trunk,
   "l2"->{PaddingLayer[{{0,0},{0,1},{0,1}}], ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]},
   "l3"->{ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp},
   "decode1"->decode1,
   "decode2"->decode2,
   "decode3"->decode3

},{
   "l1"->NetPort[{"decode1","internal"}],
   "l2"->NetPort[{"decode2","internal"}],
   "l1"->"l2"->"l3"->NetPort[{"decode3","internal"}],

   NetPort["conditional1"]->NetPort[{"decode1","conditional"}],
   NetPort["conditional2"]->NetPort[{"decode2","conditional"}],
   NetPort["conditional3"]->NetPort[{"decode3","conditional"}],
   NetPort[{"decode1","Output"}]->NetPort["Small"],
   NetPort[{"decode2","Output"}]->NetPort["Medium"],
   NetPort[{"decode3","Output"}]->NetPort["Large"]
},
   "Input"->NetEncoder[{"Image",{640,480},"ColorSpace"->"RGB"}]];


(* Works well not quite right train/valid split but in practice works nicely with validation error .001 *)


size[ face_ ] := (face[[2,1]]-face[[1,1]])


CZReplacePart[array_,rules_] := ReplacePart[ array, Select[ rules, #[[1,1]] > 0 && #[[1,2]] > 0 & ] ]


CZCentroidsToArray[ centroids_, inputDims_, arrayDims_, stride_, offset_ ] :=
   CZReplacePart[ ConstantArray[ 0, arrayDims ], kt=Map[{Ceiling[(1+inputDims[[2]]-#[[2]])/stride+offset],(1+Floor[(#[[1]]-1)/stride-offset])}->1&,centroids] ]


encode[rects_]:=(

   s1 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]<108&], { 640, 480 }, { 15, 20 }, 32, 0 ];
   m1 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]>=108&&size[#]<=155&], { 640, 480 }, { 8, 10 }, 64, 0 ];
   l1 = CZCentroidsToArray[ RegionCentroid/@Select[rects,size[#]>155&], { 640, 480 }, { 8, 10 }, 64, 0 ];
   cond1s = Transpose[ImageData[ImageFilter[Delete[Flatten[#],5]&,Image[s1],1]],{2,3,1}];
   cond1 = Append[cond1s,(First@ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"][{m1}])[[;;15]]];
   cond2s = Transpose[ImageData[ImageFilter[Delete[Flatten[#],5]&,Image[m1],1]],{2,3,1}];
   cond2 = Append[Join[cond2s, Transpose[Map[Flatten,Partition[s1,{2,2},2,1,0],{2}], {2,3,1} ] ], l1];
   cond3s = Transpose[ImageData[ImageFilter[Delete[Flatten[#],5]&,Image[l1],1]],{2,3,1}];
   cond3 = Append[cond3s,m1];
   Association[
   "Small"->s1,"Medium"->m1,"Large"->l1,"conditional1"->cond1,"conditional2"->cond2,"conditional3"->cond3
]);


files1=FileNames["~/ImageDataSets/FaceScrub/ActorImages/VGA/ActorImages1/*.jpg"];
files2=FileNames["~/ImageDataSets/FaceScrub/ActorImages/VGA/ActorImages2/*.jpg"];
files3=FileNames["~/ImageDataSets/FaceScrub/ActorImages/VGA/ActorImages3/*.jpg"];
files4=FileNames["~/ImageDataSets/FaceScrub/ActorImages/VGA/ActorImages4/*.jpg"];
files5=FileNames["~/ImageDataSets/FaceScrub/ActorImages/VGA/ActorImages5/*.jpg"];
files6=FileNames["~/ImageDataSets/FaceScrub/ActorImages/VGA/ActorImages6/*.jpg"];
files7=FileNames["~/ImageDataSets/FaceScrub/ActressImages/VGA/ActressImages1/*.jpg"];
files8=FileNames["~/ImageDataSets/FaceScrub/ActressImages/VGA/ActressImages2/*.jpg"];
files9=FileNames["~/ImageDataSets/FaceScrub/ActressImages/VGA/ActressImages3/*.jpg"];
files10=FileNames["~/ImageDataSets/FaceScrub/ActressImages/VGA/ActressImages4/*.jpg"];
files11=FileNames["~/ImageDataSets/FaceScrub/ActressImages/VGA/ActressImages5/*.jpg"];
files12=FileNames["~/ImageDataSets/FaceScrub/ActressImages/VGA/ActressImages6/*.jpg"];
files=Join[files1,files2,files3,files4,files5,files6,files7,files8,files9,files10,files11,files12];


faces1=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibVGAActor1.mx"];
faces2=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibVGAActor2.mx"];
faces3=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibVGAActor3.mx"];
faces4=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibVGAActor4.mx"];
faces5=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibVGAActor5.mx"];
faces6=Import["~/ImageDataSets/FaceScrub/ActorImages/DLibVGAActor6.mx"];
faces7=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibVGAActress1.mx"];
faces8=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibVGAActress2.mx"];
faces9=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibVGAActress3.mx"];
faces10=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibVGAActress4.mx"];
faces11=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibVGAActress5.mx"];
faces12=Import["~/ImageDataSets/FaceScrub/ActressImages/DLibVGAActress6.mx"];
faces=Join[faces1,faces2,faces3,faces4,faces5,faces6,faces7,faces8,faces9,faces10,faces11,faces12];


dataset=Table[Append[encode[faces1[[k]]],"Input"->File[files1[[k]]]],{k,1,Length[files1]}];


SeedRandom[1234];rnds=RandomSample[dataset];


{trainingSet,validationSet}={rnds[[;;7000]],rnds[[7001;;]]};


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


(* ::Input:: *)
(*vis[img_,locMatrix_]:=( *)
(*r=net2[Association["Input"->img,"conditional"->Transpose[ImageData[ImageFilter[Delete[Flatten[#],5]&,Image[locMatrix],1]],{2,3,1}]]];*)
(*m=(1-locMatrix)*r;*)
(*If[Max[m]<.5,locMatrix,*)
(*l=Position[m,Max[m]];*)
(*vis[img,ReplacePart[locMatrix,l->1]]*)
(*]*)
(*)*)


(* ::Input:: *)
(*vis[img_,sofar_]:=( *)
(*s1=sofar[[1]];m1=sofar[[2]];l1=sofar[[3]];*)
(*   cond1s = Transpose[ImageData[ImageFilter[Delete[Flatten[#],5]&,Image[s1],1]],{2,3,1}];*)
(*   cond1 = Append[cond1s,(First@ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"][{m1}])[[;;15]]];*)
(*   cond2s = Transpose[ImageData[ImageFilter[Delete[Flatten[#],5]&,Image[m1],1]],{2,3,1}];*)
(*   cond2 = Append[Join[cond2s, Transpose[Map[Flatten,Partition[s1,{2,2},2,1,0],{2}], {2,3,1} ] ], l1];*)
(*   cond3s = Transpose[ImageData[ImageFilter[Delete[Flatten[#],5]&,Image[l1],1]],{2,3,1}];*)
(*   cond3 = Append[cond3s,m1];*)
(**)
(**)
(*r=net2[Association[*)
(*"Input"->img,*)
(*"conditional1"->cond1,*)
(*"conditional2"->cond2,"conditional3"->cond3]];AppendTo[sr,r];*)
(*m={(1-s1)*r["Small"],(1-m1)*r["Medium"],(1-l1)*r["Large"]};*)
(*If[Max[m]<.5,{s1,m1,l1},*)
(*l=Position[m,Max[m]];vis[img,ReplacePart[sofar,l->1]]*)
(*]*)
(*)*)


render[ result_ ] := Join[
   

   Map[
         Rectangle[{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}+{37,37}]&,
      Position[result[[1]],x_/;x>0]],
   Map[   
      Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{65,65}/2,{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{65,65}/2]&,
      Position[result[[2]],x_/;x>0]],
         
         Map[
      Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{100,100}]&,
      Position[result[[3]],x_/;x>0]]

      ];      


(* ::Input:: *)
(*HighlightImage[img,render[ vis[img,{ConstantArray[0,{16,16}],ConstantArray[0,{8,8}]}] ] ];*)


trained=NetTrain[n2,trainingSet,ValidationSet->validationSet,TrainingProgressCheckpointing->{"Directory","~/Google Drive/Personal/Computer Science/CZModels/NMSNetTraining/"},TrainingProgressReporting->File["~/Google Drive/Personal/Computer Science/CZModels/NMSNetTraining/results.csv"]];
(* .011 is ok but not great *)
