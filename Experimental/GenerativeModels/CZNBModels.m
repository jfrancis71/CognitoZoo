(* ::Package:: *)

<<"Experimental/GenerativeModels/CZDiscreteImage.m"


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


CZCreateNBConditionalModel[ outputLayerType_, lossType_ ] := NetGraph[{
   "out"->outputLayerType,
   "crossentropy"->lossType},{
   "out"->"crossentropy"->NetPort["Loss"],
   "out"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}],
   NetPort["Conditional"]->"out"
}];


CZCreateNBModel[ conditionalDims_, outputLayerType_, lossType_ ] := NetGraph[{
   "array"->ConstantArrayLayer[conditionalDims],
   "decoder"->CZCreateNBConditionalModel[ outputLayerType, lossType ]},{
   "array"->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Input"}]
}];


SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{}};


CZCreateNBModelBinaryVector[] := CZNBModel[ CZBinaryVector, CZCreateNBModel[ {784}, LogisticSigmoid, CrossEntropyLossLayer["Binary"] ] ];


CZSample[ CZNBModel[ CZBinaryVector, net_ ] ] := Module[{out=net[ConstantArray[0,{784}]]["Output"]},
   rndBinary /@ out ];


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelBinaryImage[] := CZNBModel[ CZBinaryImage, CZCreateNBModel[ {28,28}, LogisticSigmoid, CrossEntropyLossLayer["Binary"] ] ];


CZSample[ CZNBModel[ CZBinaryImage, net_ ] ] :=
   Map[ rndBinary, net[ ConstantArray[0,{28,28}]]["Output"], {2} ];


CZCreateNBModelDiscreteImage[] := CZNBModel[ CZDiscreteImage, CZCreateNBModel[ {10,28,28}, {TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]}, CrossEntropyLossLayer["Index"]  ] ];


rndMult[probs_]:=RandomChoice[probs->Range[1,10]]


CZSample[ CZNBModel[ CZDiscreteImage, net_ ] ] :=
   Map[ rndMult, net[ConstantArray[1,{28,28}]]["Output"], {2} ]/10.;


SyntaxInformation[ CZNBModel ]= {"ArgumentsPattern"->{_,_}};


CZTrain[ CZNBModel[ modelType_, net_ ], samples_ ] :=
   CZNBModel[ modelType, NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss", MaxTrainingRounds->1000 ] ];


CZLogDensity[ CZNBModel[ _, net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
