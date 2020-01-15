(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


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


MaskCrossEntropyLossLayer[mask_, crossEntropyType_ ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "th1"->ThreadingLayer[Times],
   "th2"->ThreadingLayer[Times],
   "meancrossentropy"->If[crossEntropyType=="Binary",CrossEntropyLossLayer[crossEntropyType],CZCrossEntropyLossLayer],
   "totalcrossentropy"->ElementwiseLayer[#*Length[Flatten[mask]]&]},{
   {NetPort["Input"],"mask"}->"th1"->NetPort[{"meancrossentropy","Input"}],
   {NetPort["Target"],"mask"}->"th2"->NetPort[{"meancrossentropy","Target"}],
   NetPort[{"meancrossentropy","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}];


PredictLayer[mask_, crossEntropyType_ ]:=NetGraph[{
   "reshapeConditional"->ReshapeLayer[Prepend[mask//Dimensions,1]],
   "mask"->If[
      crossEntropyType=="Binary",
      {ReshapeLayer[Prepend[mask//Dimensions,1]], MaskLayer[{mask}]},
      MaskLayer[ConstantArray[mask,{10}]]],
   "cat"->CatenateLayer[],
   "conv"->{ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{1,1},"PaddingSize"->0],Tanh,ConvolutionLayer[
         If[crossEntropyType=="Binary",1,10],{1,1}]},
   "reshapeOutput"->If[crossEntropyType=="Binary",ReshapeLayer[mask//Dimensions],
   ElementwiseLayer[#&]],
   "probs"->If[crossEntropyType=="Binary",LogisticSigmoid,SoftmaxLayer[1]]
         },{
   NetPort["Image"]->"mask",
   {"mask","reshapeConditional"}->"cat"->"conv"->"reshapeOutput"->"probs",
   NetPort["Conditional"]->"reshapeConditional"},
   "Conditional"->(mask//Dimensions)];


CZCreatePixelCNNConditionalNet[ crossEntropyType_, pixels_ ] := Module[{masks=InformationMasking[ pixels ]},tm=masks;NetGraph[Flatten@{
   Table[ "predict"<>ToString[k]->PredictLayer[ masks[[k]], crossEntropyType ],{k,Length[pixels]}],
   Table[ "loss"<>ToString[k]->MaskCrossEntropyLossLayer[ If[crossEntropyType=="Binary",pixels[[k]],Table[pixels[[k]],{10}]], crossEntropyType ], {k,Length[pixels]} ],
   "total_loss"->TotalLayer[]
   },
   {
   Table[NetPort[{"predict"<>ToString[k],"Output"}]->NetPort[{"loss"<>ToString[k],"Input"}],{k,Length[pixels]}],
   Table[NetPort["Image"]->NetPort[{"loss"<>ToString[k],"Target"}],{k,Length[pixels]}],
   Table["loss"<>ToString[k]->"total_loss",{k,Length[pixels]}],
   "total_loss"->NetPort["Loss"]
   }]];


CZCreatePixelCNNNet[ crossEntropyType_, pixels_ ] := NetGraph[{
   "global"->ConstantArrayLayer[pixels[[1]]//Dimensions],
   "condpixelcnn"->CZCreatePixelCNNConditionalNet[ crossEntropyType, pixels ] },{
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}]


SyntaxInformation[ CZPixelCNN ]= {"ArgumentsPattern"->{}};


CZCreatePixelCNNBinary[ imageDims_:{28,28} ] :=
   CZGenerativeModel[ CZPixelCNN, CZBinary[ imageDims ], Identity, CZCreatePixelCNNNet[ "Binary", PixelCNNOrdering[ imageDims ] ] ];


CZCreatePixelCNNDiscrete[ imageDims_:{28,28} ] :=
   CZGenerativeModel[ CZPixelCNN, CZDiscrete[ imageDims ], CZOneHot, CZCreatePixelCNNNet[ "Probabilities", PixelCNNOrdering[ imageDims ] ] ];


CZSampleConditionalPixelCNN[ conditionalPixelCNNNet_, inputType_[ imageDims_ ], encoder_, conditional_ ] := Module[{s=ConstantArray[If[inputType===CZBinaryImage,0,1],imageDims], pixels=PixelCNNOrdering[ imageDims ]},
   For[k=1,k<=Length[pixels],k++,
      l = NetTake[conditionalPixelCNNNet,"predict"<>ToString[k]][Association["Image"->encoder@s,"Conditional"->conditional]];
      s = If[inputType===CZBinary,CZSampleBinary,CZSampleDiscrete][l]*pixels[[k]]+s;t=s;
   ];
   s/If[inputType===CZBinaryImage,1,10]]


CZSample[ CZGenerativeModel[ CZPixelCNN, inputType_, encoder_, pixelCNNNet_ ] ] :=
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, "condpixelcnn" ], inputType, encoder, NetExtract[ pixelCNNNet, "global" ][] ];
