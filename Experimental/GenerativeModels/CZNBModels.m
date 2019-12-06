(* ::Package:: *)

(*
   Generative models all have an Input port, an Output port and a Loss port.
   Conditional Generative models are considered to be generative models, but have
   an additional Conditional port.
   The input for discrete images in the net is eg 28x28x10
   Note this isn't always convenient for within the net, so they may reformat (to 10x28x28).
   Models are in format: CZGenerativeModel[ modelType, inputType, encoder, net ]
*)


<<"Experimental/GenerativeModels/CZDiscreteImage.m"


CZGenerativeOutputLayer[ outputLayerType_, lossType_ ] := NetGraph[{
   "out"->outputLayerType,
   "crossentropy"->lossType},{
   NetPort["Conditional"]->"out"->"crossentropy"->NetPort["Loss"],
   "out"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}]
}];


CZCreateNBModel[ conditionalDims_, outputLayerType_, lossType_ ] := NetGraph[{
   "array"->ConstantArrayLayer[conditionalDims],
   "decoder"->CZGenerativeOutputLayer[ outputLayerType, lossType ]},{
   "array"->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Input"}]
}];


SyntaxInformation[ CZNBModel ]= {"ArgumentsPattern"->{}};


CZCreateNBModelBinaryVector[ inputUnits_:784 ] := CZGenerativeModel[ CZNBModel, CZBinaryVector[ inputUnits ], Identity, CZCreateNBModel[ {inputUnits}, LogisticSigmoid, CrossEntropyLossLayer["Binary"] ] ];


CZSample[ CZGenerativeModel[ CZNBModel, CZBinaryVector[ inputUnits_ ], _, net_ ] ] := CZSampleBinaryVector@net[ConstantArray[0,{inputUnits}]]["Output"];


CZCreateNBModelBinaryImage[ imageDims_:{28,28} ] := CZGenerativeModel[ CZNBModel, CZBinaryImage[ imageDims ], Identity, CZCreateNBModel[ imageDims, LogisticSigmoid, CrossEntropyLossLayer["Binary"] ] ];


CZSample[ CZGenerativeModel[ CZNBModel, CZBinaryImage[ imageDims_ ], _, net_ ] ] := CZSampleBinaryImage@net[ ConstantArray[0,imageDims]]["Output"];


CZCreateNBModelDiscreteImage[ imageDims_:{28,28} ] := CZGenerativeModel[ CZNBModel, CZDiscreteImage[ imageDims ], Identity, CZCreateNBModel[ Prepend[imageDims, 10], {TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]}, CrossEntropyLossLayer["Index"]  ] ];


CZSample[ CZGenerativeModel[ CZNBModel, CZDiscreteImage[ imageDims_ ], _, net_ ] ] := CZSampleDiscreteImage@net[ConstantArray[1,imageDims]]["Output"]/10;


CZLogDensity[ CZGenerativeModel[ CZNBModel_, _, encoder_, net_ ], sample_ ] :=
   -net[ Association["Input" -> encoder@sample ] ]["Loss"];
