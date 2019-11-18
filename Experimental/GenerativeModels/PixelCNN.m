(* ::Package:: *)

(* ::Input:: *)
(*(* Example implementation of PixelCNN.*)
(**)*)


pixels1=pixels2=pixels3=pixels4=ConstantArray[0,{1,28,28}];
pixels = {
   (pixels1[[1,1;;28;;2,1;;28;;2]]=1;pixels1),
   (pixels2[[1,2;;28;;2,2;;28;;2]]=1;pixels2),
   (pixels3[[1,2;;28;;2,1;;28;;2]]=1;pixels3),
   (pixels4[[1,1;;28;;2,2;;28;;2]]=1;pixels4)
};


masks = FoldList[ Plus, ConstantArray[ 0, { 1, 28, 28} ], pixels ];


MaskLayer[mask_]:=NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]


MaskLossLayer[mask_]:=NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "th1"->ThreadingLayer[Times],
   "th2"->ThreadingLayer[Times],
   "meancrossentropy"->CrossEntropyLossLayer["Binary"],
   "totalcrossentropy"->ElementwiseLayer[#*784.&]},{
   {NetPort["Input"],"mask"}->"th1"->NetPort[{"meancrossentropy","Input"}],
   {NetPort["Target"],"mask"}->"th2"->NetPort[{"meancrossentropy","Target"}],
   NetPort[{"meancrossentropy","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}];


PredictLayer[mask_]:=NetGraph[{
   "mask"->MaskLayer[mask],
   "cat"->CatenateLayer[],
   "conv"->{ConvolutionLayer[16,{5,5},"PaddingSize"->2],Ramp,ConvolutionLayer[1,{1,1}]}},{
   NetPort["Image"]->"mask",
   {"mask",NetPort["Glob"]}->"cat"->"conv"}];


ConditionalPixelCNN = NetGraph[Flatten@{
   "reshapeInput"->ReshapeLayer[{1,28,28}],
   "reshapeConditional"->ReshapeLayer[{1,28,28}],
   Table[{
      "conv"<>ToString[k]->PredictLayer[masks[[k]]],
      "log"<>ToString[k]->LogisticSigmoid,
      "loss"<>ToString[k]->MaskLossLayer[pixels[[k]]]},
      {k,1,Length[pixels]}],
   "loss"->TotalLayer[]
},{
   NetPort["Image"]->"reshapeInput",
   NetPort["Conditional"]->"reshapeConditional",
   Table[{
      "reshapeInput"->"conv"<>ToString[k]->"log"<>ToString[k]->"loss"<>ToString[k], "reshapeConditional"->NetPort[{"conv"<>ToString[k],"Glob"}], "reshapeInput"->NetPort[{"loss"<>ToString[k],"Target"}],"log"<>ToString[k]->NetPort["Output"<>ToString[k]]},{k,1,Length[pixels]}],
   Table["loss"<>ToString[k],{k,1,Length[pixels]}]->"loss"->NetPort["Loss"]
},
   "Image"->{28,28}];


SyntaxInformation[ PixelCNN ]= {"ArgumentsPattern"->{_}};


CreatePixelCNN[] := PixelCNN[ NetGraph[{
   "global"->ConstantArrayLayer[{28,28}],
   "condpixelcnn"->ConditionalPixelCNN},{
   NetPort["Image"]->NetPort[{"condpixelcnn","Image"}],
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}] ];


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


Train[ PixelCNN[ pixelCNNNet_ ], samples_ ] :=
   PixelCNN[ NetTrain[ pixelCNNNet, Association["Image"->#]&/@samples,
      LearningRateMultipliers->
         Flatten[Table[
         {{"condpixelcnn","conv"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


LogDensity[ PixelCNN[ pixelCNNNet_ ], sample_ ] :=
   -pixelCNNNet[ sample ][ "Loss" ]*784/9


Sample[ PixelCNN[ pixelCNNNet_ ] ] := Module[{s=ConstantArray[0,{28,28}]},
   For[k=1,k<=Length[pixels],k++,
      l = pixelCNNNet[s]["Output"<>ToString[k]][[1]];
      s = Map[rndBinary,l,{2}]*pixels[[k]][[1]]+s;t=s;
   ];
   s]


(* ::Input:: *)
(**)
