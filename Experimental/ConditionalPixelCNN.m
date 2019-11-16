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


SyntaxInformation[ GenerativePixelCNNNet ]= {"ArgumentsPattern"->{_}};


GenerativePixelCNN = GenerativePixelCNNNet[ NetGraph[{
   "global"->ConstantArrayLayer[{28,28}],
   "condpixelcnn"->ConditionalPixelCNN},{
   NetPort["Image"]->NetPort[{"condpixelcnn","Image"}],
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}] ];


ConditionalPixel = NetGraph[{
   "reshapeInput"->ReshapeLayer[{1,28,28}],
   "reshapeConditional"->ReshapeLayer[{1,28,28}],
   "log"->LogisticSigmoid,
   "loss"->CrossEntropyLossLayer["Binary"]},{
   NetPort["Image"]->"reshapeInput"->NetPort[{"loss","Target"}],
   NetPort["Conditional"]->"reshapeConditional"->"log"->NetPort[{"loss","Input"}],
   "loss"->NetPort["Loss"],
   "log"->NetPort["Output"]},
   "Image"->{28,28}
];


SyntaxInformation[ GenerativePixelNet ]= {"ArgumentsPattern"->{_}};


GenerativePixel = GenerativePixelNet[ NetGraph[{
   "global"->ConstantArrayLayer[{28,28}],
   "condpixel"->ConditionalPixel},{
   NetPort["Image"]->NetPort[{"condpixel","Image"}],
   "global"->NetPort[{"condpixel","Conditional"}]
}] ];


Train[ GenerativePixelNet[ generativePixelNet_ ] ][ examples_ ] :=
   GenerativePixelNet[ NetTrain[ generativePixelNet, Association["Image"->#]&/@examples, 
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


LogDensity[ GenerativePixelNet[ generativePixelNet_ ] ][ example_ ] :=
   -generativePixelNet[ example ][ "Loss" ]*784


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


Sample[ GenerativePixelNet[ generativePixelNet_ ] ] :=
   Map[ rndBinary, generativePixelNet[ ConstantArray[0, {28, 28 } ] ][ "Output" ][[1]], {2} ];


Train[ GenerativePixelCNNNet[ generativePixelCNNNet_ ] ][ examples_ ] :=
   GenerativePixelCNNNet[ NetTrain[ generativePixelCNNNet, Association["Image"->#]&/@examples,
      LearningRateMultipliers->
         Flatten[Table[
         {{"condpixelcnn","conv"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


LogDensity[ GenerativePixelCNNNet[ generativePixelCNNNet_ ] ][ example_ ] :=
   -generativePixelCNNNet[ example ][ "Loss" ]*784/9


Sample[ GenerativePixelCNNNet[ generativePixelCNNNet_ ] ] := Module[{s=ConstantArray[0,{28,28}]},
   For[k=1,k<=Length[pixels],k++,
      l = generativePixelCNNNet[s]["Output"<>ToString[k]][[1]];
      s = Map[rndBinary,l,{2}]*pixels[[k]][[1]]+s;t=s;
   ];
   s]


(* ::Input:: *)
(**)
