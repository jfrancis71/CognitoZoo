(* ::Package:: *)

<<"Experimental/GenerativeModels/CZDiscreteImage.m"


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


(* Note the cross entropy is calculated on final level so might want to arrange
   Input is 28x28x10 (for probabilities)
*)
MaskCrossEntropyLossLayer[mask_, crossEntropyType_ ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "th1"->ThreadingLayer[Times],
   "th2"->ThreadingLayer[Times],
   "meancrossentropy"->CrossEntropyLossLayer[crossEntropyType],
   "totalcrossentropy"->ElementwiseLayer[#*Length[Flatten[mask]]&]},{
   {NetPort["Input"],"mask"}->"th1"->NetPort[{"meancrossentropy","Input"}],
   {NetPort["Target"],"mask"}->"th2"->NetPort[{"meancrossentropy","Target"}],
   NetPort[{"meancrossentropy","Loss"}]->"totalcrossentropy"->NetPort["Loss"]
}];


PredictLayer[mask_, crossEntropyType_ ]:=NetGraph[{
   "reshapeConditional"->ReshapeLayer[Prepend[mask//Dimensions,1]],
   "mask"->If[
      crossEntropyType=="Binary",
      {ReshapeLayer[{1,28,28}], MaskLayer[{mask}]},
      {TransposeLayer[{1<->3,2<->3}],MaskLayer[ConstantArray[mask,{10}]]}],
   "cat"->CatenateLayer[],
   "conv"->{ConvolutionLayer[16,{5,5},"PaddingSize"->2],Ramp,ConvolutionLayer[
         If[crossEntropyType=="Binary",1,10],{1,1}]},
   "reshapeOutput"->If[crossEntropyType=="Binary",ReshapeLayer[mask//Dimensions],
   TransposeLayer[{1<->3,1<->2}]],
   "probs"->If[crossEntropyType=="Binary",LogisticSigmoid,SoftmaxLayer[]]
         },{
   NetPort["Image"]->"mask",
   {"mask","reshapeConditional"}->"cat"->"conv"->"reshapeOutput"->"probs",
   NetPort["Conditional"]->"reshapeConditional"},
   "Conditional"->(mask//Dimensions)];


CZCreatePixelCNNConditionalNet[ crossEntropyType_, pixels_ ] := Module[{masks=InformationMasking[ pixels ]},NetGraph[Flatten@{
   Table[ "predict"<>ToString[k]->PredictLayer[ masks[[k]], crossEntropyType ],{k,1,4}],
   Table[ "loss"<>ToString[k]->MaskCrossEntropyLossLayer[ If[crossEntropyType=="Binary",pixels[[k]],Transpose[Table[pixels[[k]],{10}],{3,1,2}]], crossEntropyType ], {k,1,4} ],
   Table[ "maskoutput"<>ToString[k]->MaskLayer[ If[crossEntropyType=="Binary",pixels[[k]],Transpose[Table[pixels[[k]],{10}],{3,1,2}]] ], {k,1,4}],
   "total_output"->TotalLayer[],
   "total_loss"->TotalLayer[]
   },
   {
   Table[NetPort[{"predict"<>ToString[k],"Output"}]->{"maskoutput"<>ToString[k],NetPort[{"loss"<>ToString[k],"Input"}]},{k,1,4}],
   Table["maskoutput"<>ToString[k]->"total_output",{k,1,4}],
   Table[NetPort["Image"]->NetPort[{"loss"<>ToString[k],"Target"}],{k,1,4}],
   Table["loss"<>ToString[k]->"total_loss",{k,1,4}],
   "total_loss"->NetPort["Loss"]
   }]];


CZCreatePixelCNNNet[ crossEntropyType_, pixels_ ] := NetGraph[{
   "global"->ConstantArrayLayer[pixels[[1]]//Dimensions],
   "condpixelcnn"->CZCreatePixelCNNConditionalNet[ crossEntropyType, pixels ] },{
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}]


SyntaxInformation[ CZPixelCNN ]= {"ArgumentsPattern"->{}};


CZCreatePixelCNNBinaryImage[ imageDims_:{28,28} ] :=
   CZGenerativeModel[ CZPixelCNN, CZBinaryImage[ imageDims ], Identity, CZCreatePixelCNNNet[ "Binary", PixelCNNOrdering[ imageDims ] ] ];


CZCreatePixelCNNDiscreteImage[ imageDims_:{28,28} ] :=
   CZGenerativeModel[ CZPixelCNN, CZDiscreteImage[ imageDims ], CZOneHot, CZCreatePixelCNNNet[ "Probabilities", PixelCNNOrdering[ imageDims ] ] ];


CZLogDensity[ CZGenerativeModel[ CZPixelCNN, _, encoder_, net_ ], sample_ ] := net[ Association["Image"->encoder@sample ] ]["Loss"];


CZTrain[ CZGenerativeModel[ CZPixelCNN, inputType_[ imageDims_ ], encoder_, pixelCNNNet_ ], samples_ ] := Module[{pixels=PixelCNNOrdering[ imageDims ]},
   CZGenerativeModel[ CZPixelCNN, inputType[ imageDims ], encoder, NetTrain[ pixelCNNNet, Association["Image"->encoder@#]&/@samples,
      LearningRateMultipliers->
         Flatten[Table[
         {{"condpixelcnn","predict"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
]];


CZSample[ CZGenerativeModel[ CZPixelCNN, CZBinaryImage[ imageDims_ ], encoder_, pixelCNNNet_ ] ] := Module[{s=ConstantArray[0,imageDims], pixels=PixelCNNOrdering[ imageDims ]},
   For[k=1,k<=4,k++,
      l = pixelCNNNet[s]["Output"<>ToString[k]];
      s = CZSampleBinaryImage[l]*pixels[[k]]+s;t=s;
   ];
   s]


CZSample[ CZGenerativeModel[ CZPixelCNN, CZDiscreteImage[ imageDims_ ], encoder_, pixelCNNNet_ ] ] := Module[{s=ConstantArray[1,imageDims], pixels=PixelCNNOrdering[ imageDims ]},
   For[k=1,k<=4,k++,
      l = pixelCNNNet[encoder@s]["Output"<>ToString[k]];
      s = CZSampleDiscreteImage[l]*pixels[[k]]+s;t=s;
   ];
   s/10]
