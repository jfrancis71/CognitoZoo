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
   "decoder"->CZNBConditionalModelBinaryVectorNet},{
   "array"->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Input"}]
}];


SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{}};


CZCreateNBModelBinaryVector[] := CZNBModel[ CZBinaryVector, CZNBModelBinaryVectorNet ];


CZSample[ CZNBModel[ CZBinaryVector, net_ ] ] := Module[{out=net[ConstantArray[0,{784}]]["Output"]},
   rndBinary /@ out ];


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


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelBinaryImage[] := CZNBModel[ CZBinaryImage, CZNBModelBinaryImageNet ];


CZSample[ CZNBModel[ CZBinaryImage, net_ ] ] :=
   Map[ rndBinary, net[ ConstantArray[0,{28,28}]]["Output"], {2} ];


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


SyntaxInformation[ CZDiscreteImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelDiscreteImage[] := CZNBModel[ CZDiscreteImage, CZNBModelDiscreteImageNet ];


rndMult[probs_]:=RandomChoice[probs->Range[1,10]]


CZSample[ CZNBModel[ CZDiscreteImage, net_ ] ] :=
   Map[ rndMult, net[ConstantArray[1,{28,28}]]["Output"], {2} ]/10.;


SyntaxInformation[ CZNBModel ]= {"ArgumentsPattern"->{_,_}};


CZTrain[ CZNBModel[ modelType_, net_ ], samples_ ] :=
   CZNBModel[ modelType, NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss", MaxTrainingRounds->1000 ] ];


CZLogDensity[ CZNBModel[ _, net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
