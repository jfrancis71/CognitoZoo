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


GTKernel=Delete[Flatten[Table[{ReplacePart[ConstantArray[0,{3,3}],{i,j}->1]},{i,1,3},{j,1,3}],1],5];


GTKernel1=Flatten[Table[{ReplacePart[ConstantArray[0,{2,2}],{i,j}->1]},{i,1,2},{j,1,2}],1];


decode1 = NetGraph[{
   "cond1"->{ReshapeLayer[{1,15,20}],ConvolutionLayer[8,{3,3},"PaddingSize"->1,"Weights"->GTKernel]},
   "cond2"->{ReshapeLayer[{1,8,10}],ResizeLayer[{Scaled[2],Scaled[2]},"Resampling"->"Nearest"],PartLayer[{All,;;15,All}]},
"j"->CatenateLayer[],
   "c1"->ConvolutionLayer[16,{1,1}],"r1"->Ramp,
   "c2"->ConvolutionLayer[1,{1,1}],"f"->PartLayer[1],"o"->LogisticSigmoid
},{
   "j"->"c1"->"r1"->"c2"->"f"->"o",
   NetPort["conditional1"]->"cond1",
   NetPort["conditional2"]->"cond2",
   {NetPort["internal"],"cond1","cond2"}->"j"
},"internal"->{256,15,20},"conditional1"->{15,20},"conditional2"->{8,10}];


decode2 = NetGraph[{
   "cond1"->{ReshapeLayer[{1,15,20}],ConvolutionLayer[4,{2,2},"Stride"->2,"PaddingSize"->{{0,1},{0,1}},"Weights"->GTKernel1]},
   "cond2"->{ReshapeLayer[{1,8,10}],ConvolutionLayer[8,{3,3},"PaddingSize"->1,"Weights"->GTKernel]},
   "cond3"->ReshapeLayer[{1,8,10}],
"j"->CatenateLayer[],
   "c1"->ConvolutionLayer[16,{1,1}],"r1"->Ramp,
   "c2"->ConvolutionLayer[1,{1,1}],"f"->PartLayer[1],"o"->LogisticSigmoid
},{
   "j"->"c1"->"r1"->"c2"->"f"->"o",
   NetPort["conditional1"]->"cond1",
   NetPort["conditional2"]->"cond2",
   NetPort["conditional3"]->"cond3",
   {NetPort["internal"],"cond1","cond2","cond3"}->"j"
},"internal"->{256,8,10},"conditional1"->{15,20},"conditional2"->{8,10},"conditional3"->{8,10}];


decode3 = NetGraph[{
   "cond2"->ReshapeLayer[{1,8,10}],
   "cond3"->{ReshapeLayer[{1,8,10}],ConvolutionLayer[8,{3,3},"PaddingSize"->1,"Weights"->GTKernel]},
"j"->CatenateLayer[],
   "c1"->ConvolutionLayer[16,{1,1}],"r1"->Ramp,
   "c2"->ConvolutionLayer[1,{1,1}],"f"->PartLayer[1],"o"->LogisticSigmoid
},{
   "j"->"c1"->"r1"->"c2"->"f"->"o",
   NetPort["conditional2"]->"cond2",
   NetPort["conditional3"]->"cond3",
   {NetPort["internal"],"cond2","cond3"}->"j"
},"internal"->{256,8,10},"conditional2"->{8,10},"conditional3"->{8,10}];


(* Note very important, for training each of the ground truth's decoder conv for each layer should have
learning rate set to 0
   We don't want to learn ground truth = ground truth, that is trivial and uninteresting.
*)
nmsNet = NetGraph[{
   "trunk"->{trunk,BatchNormalizationLayer[]},
   "block2"->{PaddingLayer[{{0,0},{0,1},{0,1}}], ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp,PoolingLayer[{2,2},"Stride"->2]},
   "block3"->{BatchNormalizationLayer[],ConvolutionLayer[256,{3,3},"PaddingSize"->1],Ramp},
   "decode1"->decode1,
   "decode2"->decode2,
   "decode3"->decode3

},{
   "trunk"->NetPort[{"decode1","internal"}],
   "block2"->NetPort[{"decode2","internal"}],
   "trunk"->"block2"->"block3"->NetPort[{"decode3","internal"}],

   NetPort["GTFaceArray1"]->{NetPort[{"decode1","conditional1"}],NetPort[{"decode2","conditional1"}]},
   NetPort["GTFaceArray2"]->{NetPort[{"decode1","conditional2"}],NetPort[{"decode2","conditional2"}],NetPort[{"decode3","conditional2"}]},
   NetPort["GTFaceArray3"]->{NetPort[{"decode2","conditional3"}],NetPort[{"decode3","conditional3"}]},
   NetPort[{"decode1","Output"}]->NetPort["FaceArray1"],
   NetPort[{"decode2","Output"}]->NetPort["FaceArray2"],
   NetPort[{"decode3","Output"}]->NetPort["FaceArray3"]
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
   "FaceArray1"->s1,"FaceArray2"->m1,"FaceArray3"->l1,"GTFaceArray1"->s1,"GTFaceArray2"->m1,"GTFaceArray3"->l1
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


dataset=Table[Append[encode[faces[[k]]],"Input"->File[files[[k]]]],{k,1,Length[files]}];


SeedRandom[1234];rnds=RandomSample[dataset];


{trainingSet,validationSet}={rnds[[;;85000]],rnds[[85001;;]]};


(* Export["~/Google Drive/Personal/Computer Science/CZModels/CountNet2.wlnet",trained]*)


vis[img_,sofar_]:=(
s1=sofar[[1]];m1=sofar[[2]];l1=sofar[[3]];
r=net2[Association[
"Input"->img,
"GTFaceArray1"->s1,
"GTFaceArray2"->m1,"GTFaceArray3"->l1]];AppendTo[sr,r];
m={(1-s1)*r["FaceArray1"],(1-m1)*r["FaceArray2"],(1-l1)*r["FaceArray3"]};
If[Max[m]<.5,{s1,m1,l1},
l=Position[m,Max[m]];vis[img,ReplacePart[sofar,l->1]]
]
)


render[ result_ ] := Join[
   

   Map[
         Rectangle[{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}-{37,37},{32*(#[[2]]-.5),480-32*(#[[1]]-.5)}+{37,37}]&,
      Position[result[[1]],x_/;x>0]],
   Map[   
      Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{65,65},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{65,65}]&,
      Position[result[[2]],x_/;x>0]],
         
         Map[
      Rectangle[{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}-{100,100},{64*(#[[2]]-.5),480-64*(#[[1]]-.5)}+{100,100}]&,
      Position[result[[3]],x_/;x>0]]

      ];      


(* ::Input:: *)
(*HighlightImage[img,render[ vis[img,{ConstantArray[0,{15,20}],ConstantArray[0,{8,10}],ConstantArray[0,{8,10}]} ] ] ];*)


(*
trained = NetTrain[
   nmsNet,
   trainingSet,
   ValidationSet->validationSet,
   LearningRateMultipliers\[Rule]{{"decode1","cond1",2,"Weights"}\[Rule]None,{"decode2","cond1",2,"Weights"}\[Rule]None,{"decode2","cond2",2,"Weights"}\[Rule]None,{"decode3","cond3",2,"Weights"}\[Rule]None},
   TrainingProgressCheckpointing->{"Directory","~/Google Drive/Personal/Computer Science/CZModels/NMSNetTraining/"},
   TrainingProgressReporting\[Rule]{File["~/Google Drive/Personal/Computer Science/CZModels/NMSNetTraining/results.csv"],"Interval"\[Rule]Quantity[20,"Minutes"]}];
*)
(* validation loss .0047 works well *)
