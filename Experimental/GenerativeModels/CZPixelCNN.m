(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


CZLatentModelQ[ CZPixelCNN ] := False;


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


CZMaskLossLayer[ mask_, CZBinary[ dims_ ] ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "activation"->{PartLayer[1],LogisticSigmoid},
   "masked_input"->ThreadingLayer[Times],
   "masked_target"->ThreadingLayer[Times],
   "crossentropylayer"->CrossEntropyLossLayer["Binary"],
   "totalcrossentropy"->ElementwiseLayer[#*Length[Flatten[mask]]&]},{
   NetPort["Input"]->"activation",
   {"mask","activation"}->"masked_input"->NetPort[{"crossentropylayer","Input"}],
   {"mask",NetPort["Target"]}->"masked_target"->NetPort[{"crossentropylayer","Target"}],
   NetPort[{"crossentropylayer","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}] 


CZMaskLossLayer[ mask_, CZDiscrete[ dims_ ] ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->ConstantArray[mask,{10}]],
   "activation"->{SoftmaxLayer[1]},
   "masked_input"->ThreadingLayer[Times],
   "masked_target"->ThreadingLayer[Times],
   "crossentropylayer"->CZCrossEntropyLossLayer,
   "totalcrossentropy"->ElementwiseLayer[#*Length[Flatten[mask]]&]},{
   NetPort["Input"]->"activation",
   {"mask","activation"}->"masked_input"->NetPort[{"crossentropylayer","Input"}],
   {"mask",NetPort["Target"]}->"masked_target"->NetPort[{"crossentropylayer","Target"}],
   NetPort[{"crossentropylayer","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}] 


CZMaskLossLayer[ mask_, CZRealGauss[ dims_ ] ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "loss"->CZRawGaussianLossLayer,
   "th"->ThreadingLayer[Times],
   "totalcrossentropy"->SummationLayer[]},{
   NetPort["Input"]->NetPort[{"loss","Input"}],
   NetPort["Target"]->NetPort[{"loss","Target"}],
   {"mask",NetPort[{"loss","Loss"}]}->"th"->"totalcrossentropy"->NetPort["Loss"]
}];


PredictLayer[ hideMask_, type_ ] := Module[{ inputDepth = If[Head[type]===CZDiscrete,10,1] },
   NetGraph[{
   "reshapeConditional"->ReshapeLayer[Prepend[hideMask//Dimensions,1]],
   "masked_input"->{ReshapeLayer[{inputDepth,type[[1,1]],type[[1,2]]}],
      MaskLayer[ConstantArray[hideMask,{inputDepth}]]},
   "cat"->CatenateLayer[],
   "conv"->{ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{1,1},"PaddingSize"->0],Tanh,ConvolutionLayer[
         CZDistributionParameters[ type ],{1,1}]}
         },{
   NetPort["Input"]->"masked_input",
   {"masked_input","reshapeConditional"}->"cat"->"conv",
   NetPort["Conditional"]->"reshapeConditional"},
   "Conditional"->(hideMask//Dimensions)]
];


CZCreatePixelCNNConditionalNet[ crossEntropyType_, computeOrdering_ ] := Module[{hideMasks=InformationMasking[ computeOrdering ]},tm=masks;NetGraph[Flatten@{
   Table[ "predict"<>ToString[k]->PredictLayer[ hideMasks[[k]], crossEntropyType ],{k,Length[computeOrdering]}],
   Table[ "loss"<>ToString[k]->CZMaskLossLayer[ computeOrdering[[k]], crossEntropyType ],{k,Length[computeOrdering]}],
   "total_loss"->TotalLayer[]
   },
   {
   Table[NetPort["Input"]->NetPort[{"loss"<>ToString[k],"Target"}],{k,Length[computeOrdering]}],
   Table["predict"<>ToString[k]->"loss"<>ToString[k],{k,Length[computeOrdering]}],
   Table["loss"<>ToString[k]->"total_loss",{k,Length[computeOrdering]}],
   "total_loss"->NetPort["Loss"]
   }]];


CZMaskLossLayer[ PixelCNNOrdering[{28,28}][[1]], CZRealGauss[ {28,28} ] ]


CZCreatePixelCNNNet[ crossEntropyType_, pixels_ ] := NetGraph[{
   "global"->ConstantArrayLayer[pixels[[1]]//Dimensions],
   "condpixelcnn"->CZCreatePixelCNNConditionalNet[ crossEntropyType, pixels ] },{
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}]


SyntaxInformation[ CZPixelCNN ]= {"ArgumentsPattern"->{}};


CZCreatePixelCNN[ type_:CZBinary[{28,28}] ] :=
   CZGenerativeModel[ CZPixelCNN, type, CZCreatePixelCNNNet[ type, PixelCNNOrdering[ type[[1]] ] ] ];


CZSampleConditionalPixelCNN[ conditionalPixelCNNNet_, inputType_[ imageDims_ ], conditional_ ] := Module[{s=ConstantArray[If[inputType===CZBinaryImage,0,1],imageDims], pixels=PixelCNNOrdering[ imageDims ]},
   For[k=1,k<=Length[pixels],k++,
      l = NetTake[conditionalPixelCNNNet,"predict"<>ToString[k]][Association["Input"->CZEncoder[ inputType[ imageDims ] ]@s,"Conditional"->conditional]];
      s = CZSampleDistribution[ inputType[ imageDims ], l]*pixels[[k]]+s*(1-pixels[[k]]);t=s;
   ];
   s/If[inputType===CZDiscrete,10,1]]


CZSample[ CZGenerativeModel[ CZPixelCNN, inputType_, pixelCNNNet_ ] ] :=
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, "condpixelcnn" ], inputType, NetExtract[ pixelCNNNet, "global" ][] ];


CZModelLRM[ CZPixelCNN ] := Flatten[Table[
   {{"condpixelcnn","predict"<>ToString[k],"masked_input"}->0,
   {"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,4}],1];
