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


PixelCNN=NetGraph[{
   "glob1"->ConstantArrayLayer[{1,28,28}],
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
   "loss4"->MaskLossLayer[pixels4]
},{
   "glob1"->"log1"->NetPort[{"loss1","Input"}],
   NetPort["Input"]->NetPort[{"loss1","Target"}],
   NetPort["Input"]->NetPort[{"conv2","Image"}],NetPort[{"conv2","Output"}]->"log2"->"loss2",
   "glob1"->NetPort[{"conv2","Glob"}],
   NetPort["Input"]->NetPort[{"loss2","Target"}],
   NetPort["Input"]->"conv3"->"log3"->"loss3",
   "glob1"->NetPort[{"conv3","Glob"}],
   NetPort["Input"]->NetPort[{"loss3","Target"}],
   NetPort["Input"]->"conv4"->"log4"->"loss4",
   "glob1"->NetPort[{"conv4","Glob"}],
   NetPort["Input"]->NetPort[{"loss4","Target"}]
},
   "Input"->{1,28,28}];


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


(* ::Input:: *)
(*(* Example training:*)
(*im1={#}&/@ImageData/@Binarize/@ResourceData["MNIST","TrainingData"][[1;;1000,1]];*)
(*   trained=NetTrain[ PixelCNN, Association["Input"\[Rule]#]&/@im1, LearningRateMultipliers\[Rule]{*)
(*{"loss1","mask"}\[Rule]0,*)
(*{"conv2","mask"}\[Rule]0,{"loss2","mask"}\[Rule]0,*)
(*{"conv3","mask"}\[Rule]0,{"loss3","mask"}\[Rule]0,*)
(*{"conv4","mask"}\[Rule]0,{"loss4","mask"}\[Rule]0*)
(*},*)
(*MaxTrainingRounds\[Rule]10000*)
(*];*)
(**)*)


(* ::Input:: *)
(*(* Example use:*)
(**)
(*decoder1=NetTake[trained,{"glob1","log1"}];*)
(*decoder2=NetTake[trained,{"conv2","log2"}];*)
(*decoder3=NetTake[trained,{"conv3","log3"}];*)
(*decoder4=NetTake[trained,{"conv4","log4"}];*)
(*sample[]:=( *)
(*l1=decoder1[][[1]];*)
(*s1=Map[rndBinary,l1,{2}]*pixels1[[1]];*)
(*l2=decoder2[{s1}][[1]];*)
(*s2=Map[rndBinary,l2,{2}]*pixels2[[1]]+s1;*)
(*l3=decoder3[{s2}][[1]];*)
(*s3=Map[rndBinary,l3,{2}]*pixels3[[1]]+s2;*)
(*l4=decoder4[{s3}][[1]];*)
(*s4=Map[rndBinary,l4,{2}]*pixels4[[1]]+s3;*)
(*s4*)
(*);*)
(**)*)


(* ::Input:: *)
(**)
