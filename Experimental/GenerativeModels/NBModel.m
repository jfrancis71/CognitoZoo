(* ::Package:: *)

rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


CZNBConditionalModelBinaryVectorNet = NetGraph[{
   "log"->LogisticSigmoid,
   "crossentropy"->CrossEntropyLossLayer["Binary"]},{
   "log"->"crossentropy"->NetPort["Loss"],
   "log"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}],
   NetPort["Conditional"]->"log"
}];


CZNBModelBinaryVectorNet = NetGraph[{
   "array"->ConstantArrayLayer[{784}],
   "decoder"->NBConditionalModelBinaryVectorNet},{
   "array"->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Input"}]
}];


SyntaxInformation[ CZNBModelBinaryVector ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelBinaryVector[] := CZNBModelBinaryVector[ CZNBModelBinaryVectorNet ];


CZSample[ NBModelBinaryVector[ net_ ] ] := Module[{out=net[ConstantArray[0,{784}]]["Output"]},
   rndBinary /@ out ];


CZTrain[ CZNBModelBinaryVector[ net_ ], samples_ ] :=
   CZNBModelBinaryVector[ NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss" ] ];


CZLogDensity[ CZNBModelBinaryVector[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];


CZNBConditionalModelBinaryImageNet = NetGraph[{
   "log"->LogisticSigmoid,
   "crossentropy"->CrossEntropyLossLayer["Binary"]},{
   NetPort["Conditional"]->"log"->{NetPort[{"crossentropy","Input"}],NetPort["Output"]},
   NetPort["Input"]->NetPort[{"crossentropy","Target"}],
   NetPort[{"crossentropy","Loss"}]->NetPort["Loss"]
}];


CZNBModelBinaryImageNet = NetGraph[{
   "cond"->CZNBConditionalModelBinaryImageNet,
   "array"->ConstantArrayLayer[{28,28}]},{
   "array"->NetPort[{"cond","Conditional"}],
   NetPort[{"cond","Loss"}]->NetPort["Loss"]
}];


SyntaxInformation[ CZNBModelBinaryImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelBinaryImage[] := CZNBModelBinaryImage[ CZNBModelBinaryImageNet ];


CZSample[ CZNBModelBinaryImage[ net_ ] ] :=
   Map[ rndBinary, net[ ConstantArray[0,{28,28}]]["Output"], {2} ];


CZTrain[ CZNBModelBinaryImage[ net_ ], samples_ ] :=
   CZNBModelBinaryImage[ NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss" ] ];


CZLogDensity[ CZNBModelBinaryImage[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];


CZNBConditionalModelDiscreteImageNet = NetGraph[{
   "softmax"->{TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]},
   "crossentropyloss"->CrossEntropyLossLayer["Index"]
},{
   NetPort["Conditional"]->"softmax",
   NetPort["Input"]->NetPort[{"crossentropyloss","Target"}],
   NetPort[{"softmax","Output"}]->{NetPort[{"crossentropyloss","Input"}],NetPort["Output"]},
   NetPort[{"crossentropyloss","Loss"}]->NetPort["Loss"]
}];


CZNBModelDiscreteImageNet = NetGraph[{
   "const"->ConstantArrayLayer[{10,28,28}],
   "cond"->CZNBConditionalModelDiscreteImageNet
},{
   "const"->NetPort[{"cond","Conditional"}],
   NetPort["Input"]->NetPort[{"cond","Input"}]
}];


CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


SyntaxInformation[ CZNBModelDiscreteImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelDiscreteImage[] := CZNBModelDiscreteImage[ CZNBModelDiscreteImageNet ];


rndMult[probs_]:=RandomChoice[probs->Range[1,10]]


CZSample[ CZNBModelDiscreteImage[ net_ ] ] :=
   Map[ rndMult, net[ConstantArray[1,{28,28}]]["Output"], {2} ]/10.;


CZTrain[ CZNBModelDiscreteImage[ net_ ], samples_ ] :=
   CZNBModelDiscreteImage[ NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss", MaxTrainingRounds->1000 ] ];


CZLogDensity[ CZNBModelDiscreteImage[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
