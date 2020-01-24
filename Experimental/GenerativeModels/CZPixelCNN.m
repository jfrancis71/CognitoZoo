(* ::Package:: *)

(*
   Conditional Image Generation with PixelCNN Decoders, 2016
   van den Oord, Kalchbrenner, Vinyals, Espeholt, Graves, Kavukcuoglu
   http: https://arxiv.org/pdf/1606.05328.pdf
*)


<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


CZLatentModelQ[ CZPixelCNN ] := False;


PixelCNNOrdering2D[ imageDims_ ] := Module[{pixels=ConstantArray[0,Prepend[imageDims,4]]},{
   (pixels[[1,1;;imageDims[[1]];;2,1;;imageDims[[2]];;2]]=1;pixels[[1]]),
   (pixels[[2,2;;imageDims[[1]];;2,2;;imageDims[[2]];;2]]=1;pixels[[2]]),
   (pixels[[3,2;;imageDims[[1]];;2,1;;imageDims[[2]];;2]]=1;pixels[[3]]),
   (pixels[[4,1;;imageDims[[1]];;2,2;;imageDims[[2]];;2]]=1;pixels[[4]])
}];


PixelCNNOrder[ dims_ ] := If[Length[dims]==2,
   PixelCNNOrdering2D[ dims ],
   Flatten[Table[ block=ConstantArray[0,dims];
      block[[c]]=PixelCNNOrdering2D[ dims[[2;;3]] ][[sp]];block, {sp,1,4},{c,dims[[1]]}],1]]


InformationMasking[ pixelCNNOrdering_ ] := FoldList[ Plus, pixelCNNOrdering[[1]]*0.0, pixelCNNOrdering ];


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
}]


PredictLayer[ hideMask_, type_ ] := Module[{ inputDepth = If[Head[type]===CZDiscrete,10,1] },
   NetGraph[{
   "reshapeConditional"->ReshapeLayer[Prepend[type[[1,-2;;]],1]],
   "masked_input"->{MaskLayer[hideMask],ReshapeLayer[type[[1]]]
      },
   "cat"->CatenateLayer[],
   "conv"->{ConvolutionLayer[16,{3,3},"PaddingSize"->1],Tanh,ConvolutionLayer[16,{1,1},"PaddingSize"->0],Tanh,ConvolutionLayer[
         CZDistributionParameters[ type ]*type[[1,1]],{1,1}]},
   "reshape"->ReshapeLayer[Prepend[ type[[1]], CZDistributionParameters[ type ] ] ]
         },{
   NetPort["Input"]->"masked_input",
   {"masked_input","reshapeConditional"}->"cat"->"conv"->"reshape",
   NetPort["Conditional"]->"reshapeConditional"},
   "Conditional"->type[[1,-2;;]]]
];


CZCreatePixelCNNConditionalNet[ crossEntropyType_, computeOrdering_ ] := Module[{hideMasks=InformationMasking[ computeOrdering ]},tm=masks;NetGraph[Flatten@{
   Table[ "predict"<>ToString[k]->PredictLayer[ hideMasks[[k]], crossEntropyType ],{k,Length[computeOrdering]}],
   Table[ "loss"<>ToString[k]->CZMaskLossLayer[ computeOrdering[[k]], crossEntropyType ],{k,Length[computeOrdering]}],
   "total_loss"->TotalLayer[]
   },
   {
   Table[NetPort["Target"]->NetPort[{"loss"<>ToString[k],"Target"}],{k,Length[computeOrdering]}],
   Table["predict"<>ToString[k]->"loss"<>ToString[k],{k,Length[computeOrdering]}],
   Table["loss"<>ToString[k]->"total_loss",{k,Length[computeOrdering]}],
   "total_loss"->NetPort["Loss"]
   }]];


CZCreatePixelCNNNet[ crossEntropyType_, pixels_ ] := NetGraph[{
   "global"->ConstantArrayLayer[crossEntropyType[[1,-2;;]]],
   "condpixelcnn"->CZCreatePixelCNNConditionalNet[ crossEntropyType, pixels ] },{
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}]


SyntaxInformation[ CZPixelCNN ]= {"ArgumentsPattern"->{}};


CZCreatePixelCNN[ type_:CZBinary[{1,28,28}] ] :=
   CZGenerativeModel[ CZPixelCNN, type, CZCreatePixelCNNNet[ type, PixelCNNOrder[ type[[1]] ] ] ];


CZSampleConditionalPixelCNN[ conditionalPixelCNNNet_, inputType_[ imageDims_ ], conditional_ ] := Module[{s=ConstantArray[If[inputType===CZBinaryImage,0,1],imageDims], pixels=PixelCNNOrder[ imageDims ]},
   For[k=1,k<=Length[pixels],k++,
      l = NetTake[conditionalPixelCNNNet,"predict"<>ToString[k]][Association["Input"->s,"Conditional"->conditional]];
      s = CZSampleDistribution[ inputType[ imageDims ], l]*pixels[[k]]+s*(1-pixels[[k]]);t=s;
   ];
   s/If[inputType===CZDiscrete,10,1]]


CZSample[ CZGenerativeModel[ CZPixelCNN, inputType_, pixelCNNNet_ ] ] :=
   CZSampleConditionalPixelCNN[ NetExtract[ pixelCNNNet, "condpixelcnn" ], inputType, NetExtract[ pixelCNNNet, "global" ][] ];


CZModelLRM[ CZPixelCNN, dims_ ] := Flatten[Table[
   {{"condpixelcnn","predict"<>ToString[k],"masked_input"}->0,
   {"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,If[Length[dims]==2,4,4*dims[[1]]]}],1];
