(* ::Package:: *)

(* ::Input:: *)
(*(* Example implementation of PixelCNN.*)
(**)*)


pixels1=ConstantArray[0,{1,28,28}];
pixels1[[1,1;;28;;2,1;;28;;2]]=1;
pixels2=ConstantArray[0,{1,28,28}];
pixels2[[1,2;;28;;2,2;;28;;2]]=1;
pixels3=ConstantArray[0,{1,28,28}];
pixels3[[1,1;;28;;2,2;;28;;2]]=1;
pixels4=ConstantArray[0,{1,28,28}];
pixels4[[1,2;;28;;2,1;;28;;2]]=1;


mask1=mask2=mask3=mask4=ConstantArray[0,{1,28,28}];
mask2=mask1+pixels1;
mask3=mask2+pixels2;
mask4=mask3+pixels3;


MaskLayer[mask_]:=NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]


MaskLossLayer[mask_]:=NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "th1"->ThreadingLayer[Times],
   "th2"->ThreadingLayer[Times],
   "meancrossentropy"->CrossEntropyLossLayer["Binary"]},{
   {NetPort["Input"],"mask"}->"th1"->NetPort[{"meancrossentropy","Input"}],
   {NetPort["Target"],"mask"}->"th2"->NetPort[{"meancrossentropy","Target"}]
}];


PredictLayer[mask_]:=NetGraph[{
   "mask"->MaskLayer[mask],
   "cat"->CatenateLayer[],
   "conv"->ConvolutionLayer[1,{3,3},"PaddingSize"->1]},{
   NetPort["Image"]->"mask",
   {"mask",NetPort["Glob"]}->"cat"->"conv"}];


ConditionalPixelCNN = NetGraph[{
   "reshapeInput"->ReshapeLayer[{1,28,28}],
   "reshapeConditional"->ReshapeLayer[{1,28,28}],
   "conv1"->PredictLayer[mask1],
   "log1"->LogisticSigmoid,
   "loss1"->MaskLossLayer[pixels1],
   "conv2"->PredictLayer[mask2],
   "log2"->LogisticSigmoid,
   "loss2"->MaskLossLayer[pixels2],
   "conv3"->PredictLayer[mask3],
   "log3"->LogisticSigmoid,
   "loss3"->MaskLossLayer[pixels3],
   "conv4"->PredictLayer[mask4],
   "log4"->LogisticSigmoid,
   "loss4"->MaskLossLayer[pixels4],
   "loss"->TotalLayer[]
},{
   NetPort["Input"]->"reshapeInput",
   NetPort["Conditional"]->"reshapeConditional",
   "reshapeInput"->"conv1"->"log1"->"loss1", "reshapeConditional"->NetPort[{"conv1","Glob"}], "reshapeInput"->NetPort[{"loss1","Target"}],"log1"->NetPort["Output1"],
   "reshapeInput"->"conv2"->"log2"->"loss2", "reshapeConditional"->NetPort[{"conv2","Glob"}], "reshapeInput"->NetPort[{"loss2","Target"}],"log2"->NetPort["Output2"],
   "reshapeInput"->"conv3"->"log3"->"loss3", "reshapeConditional"->NetPort[{"conv3","Glob"}], "reshapeInput"->NetPort[{"loss3","Target"}],"log3"->NetPort["Output3"],
   "reshapeInput"->"conv4"->"log4"->"loss4", "reshapeConditional"->NetPort[{"conv4","Glob"}], "reshapeInput"->NetPort[{"loss4","Target"}],"log4"->NetPort["Output4"],
   {"loss1","loss2","loss3","loss4"}->"loss"->NetPort["Loss"]
},
   "Input"->{28,28}];


SyntaxInformation[ GenerativePixelCNNNet ]= {"ArgumentsPattern"->{_}};


GenerativePixelCNN = GenerativePixelCNNNet[ NetGraph[{
   "global"->ConstantArrayLayer[{28,28}],
   "condpixelcnn"->ConditionalPixelCNN},{
   NetPort["Input"]->NetPort[{"condpixelcnn","Input"}],
   "global"->NetPort[{"condpixelcnn","Conditional"}]
}] ];


ConditionalPixel = NetGraph[{
   "reshapeInput"->ReshapeLayer[{1,28,28}],
   "reshapeConditional"->ReshapeLayer[{1,28,28}],
   "log"->LogisticSigmoid,
   "loss"->CrossEntropyLossLayer["Binary"]},{
   NetPort["Input"]->"reshapeInput"->NetPort[{"loss","Target"}],
   NetPort["Conditional"]->"reshapeConditional"->"log"->NetPort[{"loss","Input"}],
   "loss"->NetPort["Loss"],
   "log"->NetPort["Output"]},
   "Input"->{28,28}
];


SyntaxInformation[ GenerativePixelNet ]= {"ArgumentsPattern"->{_}};


GenerativePixel = GenerativePixelNet[ NetGraph[{
   "global"->ConstantArrayLayer[{28,28}],
   "condpixel"->ConditionalPixel},{
   NetPort["Input"]->NetPort[{"condpixel","Input"}],
   "global"->NetPort[{"condpixel","Conditional"}]
}] ];


Train[ GenerativePixelNet[ generativePixelNet_ ] ][ examples_ ] :=
   GenerativePixelNet[ NetTrain[ generativePixelNet, Association["Input"->#]&/@examples, 
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


LogDensity[ GenerativePixelNet[ generativePixelNet_ ] ][ example_ ] :=
   -generativePixelNet[ example ][ "Loss" ]


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


Sample[ GenerativePixelNet[ generativePixelNet_ ] ] :=
   Map[ rndBinary, generativePixelNet[ ConstantArray[0, {28, 28 } ] ][ "Output" ][[1]], {2} ];


Train[ GenerativePixelCNNNet[ generativePixelCNNNet_ ] ][ examples_ ] :=
   GenerativePixelCNNNet[ NetTrain[ generativePixelCNNNet, Association["Input"->#]&/@examples,
      LearningRateMultipliers->{
         {"condpixelcnn","conv1","mask"}->0,{"condpixelcnn","loss1","mask"}->0,
         {"condpixelcnn","conv2","mask"}->0,{"condpixelcnn","loss2","mask"}->0,
         {"condpixelcnn","conv3","mask"}->0,{"condpixelcnn","loss3","mask"}->0,
         {"condpixelcnn","conv4","mask"}->0,{"condpixelcnn","loss4","mask"}->0
      },
      MaxTrainingRounds->10000,LossFunction->"Loss" ]
];


LogDensity[ GenerativePixelCNNNet[ generativePixelCNNNet_ ] ][ example_ ] :=
   -generativePixelCNNNet[ example ][ "Loss" ]


Sample[ GenerativePixelCNNNet[ generativePixelCNNNet_ ] ] := (
   l1=generativePixelCNNNet[ConstantArray[0,{28,28}]]["Output1"][[1]];
   s1=Map[rndBinary,l1,{2}]*pixels1[[1]];
   l2=generativePixelCNNNet[s1]["Output2"][[1]];
   s2=Map[rndBinary,l2,{2}]*pixels2[[1]]+s1;
   l3=generativePixelCNNNet[s2]["Output3"][[1]];
   s3=Map[rndBinary,l3,{2}]*pixels3[[1]]+s2;
   l4=generativePixelCNNNet[s3]["Output4"][[1]];
   s4=Map[rndBinary,l4,{2}]*pixels4[[1]]+s3;
   s4
);


(* ::Input:: *)
(**)
