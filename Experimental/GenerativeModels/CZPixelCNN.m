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


SyntaxInformation[ CZPixelCNN ]= {"ArgumentsPattern"->{_,_}};


CZCreatePixelCNNBinaryImage[ imageDims_:{28,28} ] := CZPixelCNN[ CZBinaryImage[ imageDims ], NetGraph[{
   "global"->ConstantArrayLayer[imageDims],
   "condpixelcnn"->CZConditionalPixelCNNBinaryImage[ imageDims ]},{
   NetPort["Image"]->NetPort[{"condpixelcnn","Image"}],
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}] ]


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


CZTrain[ CZPixelCNN[ CZBinaryImage[ imageDims_ ], pixelCNNNet_ ], samples_ ] := Module[{pixels=PixelCNNOrdering[ imageDims ]},
   CZPixelCNN[ CZBinaryImage[ imageDims ], NetTrain[ pixelCNNNet, Association["Image"->#]&/@samples,
      LearningRateMultipliers->
         Flatten[Table[
         {{"condpixelcnn","conv"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
]];


CZLogDensity[ CZPixelCNN[ CZBinaryImage[ imageDims_ ], pixelCNNNet_ ], sample_ ] :=
   -pixelCNNNet[ sample ][ "Loss" ]*784/9


CZSample[ CZPixelCNN[ CZBinaryImage[ imageDims_ ], pixelCNNNet_ ] ] := Module[{s=ConstantArray[0,imageDims]},
   For[k=1,k<=Length[pixels],k++,
      l = pixelCNNNet[s]["Output"<>ToString[k]];
      s = Map[rndBinary,l,{2}]*pixels[[k]]+s;t=s;
   ];
   s]


(* We're sticking with a 1 hot encoding. Not the most efficient, but can't see a way to mask out cross entropy loss
   any other way. Can't just multiply output by zero's in mask because CrossEntropyLoss produces only one Real for all
   the inputs. On the input side, it doesn't like the use of calculated numbers, indexes need to be bounded integers.
*)
MaskLossLayerDiscrete[mask_] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->Transpose[Table[mask,{10}],{3,1,2}]],
   "th1"->ThreadingLayer[Times],
   "th2"->ThreadingLayer[Times],
   "meancrossentropy"->CrossEntropyLossLayer["Probabilities"],
   "totalcrossentropy"->ElementwiseLayer[#*784.&]},{
   {NetPort["Input"],"mask"}->"th1"->NetPort[{"meancrossentropy","Input"}],
   {NetPort["Target"],"mask"}->"th2"->NetPort[{"meancrossentropy","Target"}],
   NetPort[{"meancrossentropy","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}];


PredictLayerDiscrete[mask_] := NetGraph[{
   "mask"->{TransposeLayer[{1<->3,2<->3}],MaskLayer[ConstantArray[mask,{10}]]},
   "cat"->CatenateLayer[],
   "conv"->{ConvolutionLayer[16,{5,5},"PaddingSize"->2],Ramp,ConvolutionLayer[10,{1,1}]}},{
   NetPort["Image"]->"mask",
   {"mask",NetPort["Glob"]}->"cat"->"conv"},
   "Glob"->Prepend[mask//Dimensions,1]]


CZConditionalPixelCNNDiscreteImage[ imageDims_ ] :=  Module[{ pixels = PixelCNNOrdering[ imageDims ] }, masks = InformationMasking[ pixels ];tpix=pixels;
   NetGraph[Flatten@{
   "reshapeInput"->ReshapeLayer[Append[ imageDims, 10 ]],
   "reshapeConditional"->ReshapeLayer[Prepend[ imageDims, 1 ]],
   Table[{
      "conv"<>ToString[k]->{PredictLayerDiscrete[masks[[k]]],TransposeLayer[{3<->1,1<->2}]},
      "softmax"<>ToString[k]->SoftmaxLayer[],
      "loss"<>ToString[k]->MaskLossLayerDiscrete[pixels[[k]]]},
      {k,1,Length[pixels]}],
   "loss"->TotalLayer[]
},{
   NetPort["Image"]->"reshapeInput",
   NetPort["Conditional"]->"reshapeConditional",
   Table[{
      "reshapeInput"->"conv"<>ToString[k]->"softmax"<>ToString[k]->"loss"<>ToString[k], "reshapeConditional"->NetPort[{"conv"<>ToString[k],"Glob"}],
       NetPort["Image"]->NetPort[{"loss"<>ToString[k],"Target"}],"softmax"<>ToString[k]->NetPort["Output"<>ToString[k]]},{k,1,Length[pixels]}],
  Table["loss"<>ToString[k],{k,1,Length[pixels]}]->"loss"->NetPort["Loss"]
}]](*,
   "Image"\[Rule]Append[imageDims,10]]*)


CZCreatePixelCNNDiscreteImage[ imageDims_:{28,28} ] := CZPixelCNN[ CZDiscreteImage[ imageDims ], NetGraph[{
   "global"->ConstantArrayLayer[imageDims],
   "condpixelcnn"->CZConditionalPixelCNNDiscreteImage[ imageDims ]},{
   NetPort["Image"]->NetPort[{"condpixelcnn","Image"}],
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}] ];


CZTrain[ CZPixelCNN[ CZDiscreteImage[ imageDims_ ], pixelCNNNet_ ], samples_ ] :=
   CZPixelCNN[ CZDiscreteImage[ imageDims ], NetTrain[ pixelCNNNet, Association["Image"->CZOneHot@#]&/@samples,
      LearningRateMultipliers->
         Flatten[Table[
         {{"condpixelcnn","conv"<>ToString[k],1,"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


CZSampleConditionalPixelCNNDiscreteImage[ condPixelCNN_, imageDims_, cond_ ] := Module[{s=ConstantArray[1,imageDims]},
   pixels = PixelCNNOrdering[ imageDims ];
   For[k=1,k<=1,k++,
      l = Normal@condPixelCNN[Association["Image"->CZOneHot[s],"Conditional"->cond]]["Output"<>ToString[k]];
      s = CZSampleDiscreteImage[l]*pixels[[k]]+s;t=s;
   ];
   s/10
];


CZSample[ CZPixelCNN[ CZDiscreteImage[ imageDims_ ], pixelCNNNet_ ] ] := CZSampleConditionalPixelCNNDiscreteImage[
   NetExtract[ pixelCNNNet, "condpixelcnn" ], imageDims,
   NetExtract[ pixelCNNNet, {"global","Array"} ]
];


CZLogDensity[ CZPixelCNN[ CZDiscreteImage[ imageDims_ ], pixelCNNNet_ ], sample_ ] :=
   -pixelCNNNet[ CZOneHot@sample ][ "Loss" ]*784/9
