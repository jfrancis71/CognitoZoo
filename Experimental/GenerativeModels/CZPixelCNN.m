(* ::Package:: *)

<<"Experimental/GenerativeModels/CZDiscreteImage.m"


(* ::Input:: *)
(*(* Example implementation of PixelCNN.*)
(**)*)


PixelCNNOrdering[ imageDims_ ] := Module[{pixels=ConstantArray[0,Prepend[imageDims,4]]},{
   (pixels[[1,1;;imageDims[[1]];;2,1;;imageDims[[2]];;2]]=1;pixels[[1]]),
   (pixels[[2,2;;imageDims[[1]];;2,2;;imageDims[[2]];;2]]=1;pixels[[2]]),
   (pixels[[3,2;;imageDims[[1]];;2,1;;imageDims[[2]];;2]]=1;pixels[[3]]),
   (pixels[[4,1;;imageDims[[1]];;2,2;;imageDims[[2]];;2]]=1;pixels[[4]])
}];


InformationMasking[ pixelCNNOrdering_ ] := FoldList[ Plus, pixelCNNOrdering[[1]]*0.0, pixelCNNOrdering ];


MaskLayer[mask_]:=NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]


MaskLossLayerBinary[mask_]:=NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "th1"->ThreadingLayer[Times],
   "th2"->ThreadingLayer[Times],
   "meancrossentropy"->CrossEntropyLossLayer["Binary"],
   "totalcrossentropy"->ElementwiseLayer[#*Length[Flatten[mask]]&]},{
   {NetPort["Input"],"mask"}->"th1"->NetPort[{"meancrossentropy","Input"}],
   {NetPort["Target"],"mask"}->"th2"->NetPort[{"meancrossentropy","Target"}],
   NetPort[{"meancrossentropy","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}];


PredictLayerBinary[mask_]:=NetGraph[{
   "mask"->MaskLayer[mask],
   "cat"->CatenateLayer[],
   "conv"->{ConvolutionLayer[16,{5,5},"PaddingSize"->2],Ramp,ConvolutionLayer[1,{1,1}]},
   "reshapeMask"->ReshapeLayer[Prepend[mask//Dimensions,1]],
   "reshapeGlob"->ReshapeLayer[Prepend[mask//Dimensions,1]],
   "reshapeOutput"->ReshapeLayer[mask//Dimensions]},{
   NetPort["Image"]->"mask"->"reshapeMask",
   NetPort["Glob"]->"reshapeGlob",
   {"reshapeMask","reshapeGlob"}->"cat"->"conv"->"reshapeOutput"},"Glob"->(mask//Dimensions)];


CZConditionalPixelCNNBinaryImage[ imageDims_:{28,28} ] := Module[{ pixels = PixelCNNOrdering[ imageDims ] }, masks = InformationMasking[ pixels ];tpix=pixels;
   NetGraph[Flatten@{
      Table[{
         "conv"<>ToString[k]->PredictLayerBinary[masks[[k]]],
         "log"<>ToString[k]->LogisticSigmoid,
         "loss"<>ToString[k]->MaskLossLayerBinary[pixels[[k]]]},
         {k,1,Length[pixels]}],
      "loss"->TotalLayer[]
},{
   Table[{
      "conv"<>ToString[k]->"log"<>ToString[k]->"loss"<>ToString[k], NetPort["Conditional"]->NetPort[{"conv"<>ToString[k],"Glob"}], NetPort["Image"]->NetPort[{"loss"<>ToString[k],"Target"}],"log"<>ToString[k]->NetPort["Output"<>ToString[k]]},{k,1,Length[pixels]}],
   Table["loss"<>ToString[k],{k,1,Length[pixels]}]->"loss"->NetPort["Loss"]
},
   "Image"->imageDims]
];


SyntaxInformation[ CZPixelCNNBinaryImage ]= {"ArgumentsPattern"->{_,_}};


CZCreatePixelCNNBinaryImage[ imageDims_:{28,28} ] := CZPixelCNNBinaryImage[ imageDims, NetGraph[{
   "global"->ConstantArrayLayer[imageDims],
   "condpixelcnn"->CZConditionalPixelCNNBinaryImage[ imageDims ]},{
   NetPort["Image"]->NetPort[{"condpixelcnn","Image"}],
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}] ]


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


CZTrain[ CZPixelCNNBinaryImage[ imageDims_, pixelCNNNet_ ], samples_ ] :=
   CZPixelCNNBinaryImage[ imageDims, NetTrain[ pixelCNNNet, Association["Image"->#]&/@samples,
      LearningRateMultipliers->
         Flatten[Table[
         {{"condpixelcnn","conv"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


CZLogDensity[ CZPixelCNNBinaryImage[ imageDims_, pixelCNNNet_ ], sample_ ] :=
   -pixelCNNNet[ sample ][ "Loss" ]*784/9


CZSample[ CZPixelCNNBinaryImage[ imageDims_, pixelCNNNet_ ] ] := Module[{s=ConstantArray[0,imageDims]},
   For[k=1,k<=Length[pixels],k++,
      l = pixelCNNNet[s]["Output"<>ToString[k]];
      s = Map[rndBinary,l,{2}]*pixels[[k]]+s;t=s;
   ];
   s]


SyntaxInformation[ CZPixelCNNDiscreteImage ]= {"ArgumentsPattern"->{_}};


(* We're sticking with a 1 hot encoding. Not the most efficient, but can't see a way to mask out cross entropy loss
   any other way. Can't just multiply output by zero's in mask because CrossEntropyLoss produces only one Real for all
   the inputs. On the input side, it doesn't like the use of calculated numbers, indexes need to be bounded integers.
*)
MaskLossLayerDiscrete[mask_] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->Table[mask,{10}]],
   "th1"->ThreadingLayer[Times],
   "th2"->ThreadingLayer[Times],
   "meancrossentropy"->CrossEntropyLossLayer["Probabilities"],
   "totalcrossentropy"->ElementwiseLayer[#*784.&]},{
   {NetPort["Input"],"mask"}->"th1"->NetPort[{"meancrossentropy","Input"}],
   {NetPort["Target"],"mask"}->"th2"->NetPort[{"meancrossentropy","Target"}],
   NetPort[{"meancrossentropy","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}];


PredictLayerDiscrete[mask_] := NetGraph[{
   "mask"->MaskLayer[ConstantArray[mask,{10}]],
   "cat"->CatenateLayer[1],
   "conv"->{ConvolutionLayer[16,{5,5},"PaddingSize"->2],Ramp,ConvolutionLayer[10,{1,1}]}},{
   NetPort["Image"]->"mask",
   {"mask",NetPort["Glob"]}->"cat"->"conv"},
   "Glob"->Prepend[mask//Dimensions,1]]


CZConditionalPixelCNNDiscreteImage[ imageDims_ ] :=  Module[{ pixels = PixelCNNOrdering[ imageDims ] }, masks = InformationMasking[ pixels ];tpix=pixels;
   NetGraph[Flatten@{
   "reshapeInput"->ReshapeLayer[Prepend[ imageDims, 10 ]],
   "reshapeConditional"->ReshapeLayer[Prepend[ imageDims, 1 ]],
   Table[{
      "conv"<>ToString[k]->PredictLayerDiscrete[masks[[k]]],
      "softmax"<>ToString[k]->SoftmaxLayer[1],
      "loss"<>ToString[k]->MaskLossLayerDiscrete[pixels[[k]]]},
      {k,1,Length[pixels]}],
   "loss"->TotalLayer[]
},{
   NetPort["Image"]->"reshapeInput",
   NetPort["Conditional"]->"reshapeConditional",
   Table[{
      "reshapeInput"->"conv"<>ToString[k]->"softmax"<>ToString[k]->"loss"<>ToString[k], "reshapeConditional"->NetPort[{"conv"<>ToString[k],"Glob"}], "reshapeInput"->NetPort[{"loss"<>ToString[k],"Target"}],"softmax"<>ToString[k]->NetPort["Output"<>ToString[k]]},{k,1,Length[pixels]}],
   Table["loss"<>ToString[k],{k,1,Length[pixels]}]->"loss"->NetPort["Loss"]
},
   "Image"->Prepend[imageDims,10]]
];


CZCreatePixelCNNDiscreteImage[ imageDims_:{28,28} ] := CZPixelCNNDiscreteImage[ NetGraph[{
   "global"->ConstantArrayLayer[imageDims],
   "condpixelcnn"->CZConditionalPixelCNNDiscreteImage[ imageDims ]},{
   NetPort["Image"]->NetPort[{"condpixelcnn","Image"}],
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}] ];


CZTrain[ CZPixelCNNDiscreteImage[ pixelCNNNet_ ], samples_ ] :=
   CZPixelCNNDiscreteImage[ NetTrain[ pixelCNNNet, Association["Image"->CZOneHot@#]&/@samples,
      LearningRateMultipliers->
         Flatten[Table[
         {{"condpixelcnn","conv"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


CZSample[ CZPixelCNNDiscreteImage[ imageDims_, pixelCNNNet_ ] ] := Module[{s=ConstantArray[0,Prepend[imageDims,10]]},
   For[k=1,k<=Length[pixels],k++,
      l = pixelCNNNet[s]["Output"<>ToString[k]];
      s = discreteSample[l]*Table[pixels[[k]][[1]],{10}]+s;t=s;
   ];
   Map[
Position[#,1][[1,1]]/10.&,
Transpose[s,{3,1,2}],{2}]
   ]
