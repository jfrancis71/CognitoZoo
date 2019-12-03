(* ::Package:: *)

<<"Experimental/GenerativeModels/CZDiscreteImage.m"


CZSampleBinaryVector[ betas_ ] := RandomChoice[{1-#,#}->{0,1}]& /@ betas;


CZSampleBinaryImage[ betas_ ] := Map[ RandomChoice[{1-#,#}->{0,1}]&, betas, {2}];


CZGenerativeOutputLayer[ outputLayerType_, lossType_ ] := NetGraph[{
   "out"->outputLayerType,
   "crossentropy"->lossType},{
   "out"->"crossentropy"->NetPort["Loss"],
   "out"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}],
   NetPort["Conditional"]->"out"
}];


CZCreateNBModel[ conditionalDims_, outputLayerType_, lossType_ ] := NetGraph[{
   "array"->ConstantArrayLayer[conditionalDims],
   "decoder"->CZGenerativeOutputLayer[ outputLayerType, lossType ]},{
   "array"->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Input"}]
}];


SyntaxInformation[ CZNBModel ]= {"ArgumentsPattern"->{_,_,_}};


SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{}};


CZCreateNBModelBinaryVector[ inputUnits_:784 ] := CZNBModel[ CZBinaryVector, inputUnits, CZCreateNBModel[ {inputUnits}, LogisticSigmoid, CrossEntropyLossLayer["Binary"] ] ];


CZSample[ CZNBModel[ CZBinaryVector, inputUnits_, net_ ] ] := CZSampleBinaryVector@net[ConstantArray[0,{inputUnits}]]["Output"];


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelBinaryImage[ imageDims_:{28,28} ] := CZNBModel[ CZBinaryImage, imageDims, CZCreateNBModel[ imageDims, LogisticSigmoid, CrossEntropyLossLayer["Binary"] ] ];


CZSample[ CZNBModel[ CZBinaryImage, imageDims_, net_ ] ] := CZSampleBinaryImage@net[ ConstantArray[0,imageDims]]["Output"];


SyntaxInformation[ CZDiscreteImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelDiscreteImage[ imageDims_:{28,28} ] := CZNBModel[ CZDiscreteImage, imageDims, CZCreateNBModel[ Prepend[imageDims, 10], {TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]}, CrossEntropyLossLayer["Index"]  ] ];


CZSample[ CZNBModel[ CZDiscreteImage, imageDims_, net_ ] ] := CZSampleDiscreteImage@net[ConstantArray[1,imageDims]]["Output"]/10;


CZTrain[ CZNBModel[ modelType_, dims_, net_ ], samples_ ] :=
   CZNBModel[ modelType, dims, NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss", MaxTrainingRounds->1000 ] ];


CZLogDensity[ CZNBModel[ _, net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
