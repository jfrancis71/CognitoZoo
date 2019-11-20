(* ::Package:: *)

rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


NBModelBinaryVectorNet = NetGraph[{
   "array"->ConstantArrayLayer[{784}],
   "log"->LogisticSigmoid,
   "crossentropy"->CrossEntropyLossLayer["Binary"]},{
   "array"->"log"->"crossentropy"->NetPort["Loss"],
   "log"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}]
}];


SyntaxInformation[ NBModelBinaryVector ]= {"ArgumentsPattern"->{_}};


CreateNBModelBinaryVector[] := NBModelVector[ NBModelBinaryVectorNet ];


Sample[ NBModelBinaryVector[ net_ ] ] := Module[{out=NetTake[net,{"array","log"}]},t=out;
   rndBinary /@ out[] ];


Train[ NBModelBinaryVector[ net_ ], samples_ ] :=
   NBModelBinaryVector[ NetTrain[ net, Association[ "Input"->#, "Output"->#]&/@samples, LossFunction->"Loss" ] ];


LogDensity[ NBModelBinaryVector[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];


NBModelBinaryImageNet = NetGraph[{
   "nbmodel"->NBModelBinaryVectorNet,
   "reshapeinput"->ReshapeLayer[{784}],
   "reshapeoutput"->ReshapeLayer[{28,28}]},{
   NetPort["Input"]->"reshapeinput"->NetPort[{"nbmodel","Input"}],
   NetPort[{"nbmodel","Output"}]->"reshapeoutput"->NetPort["Output"],
   NetPort[{"nbmodel","Loss"}]->NetPort["Loss"]
}];


SyntaxInformation[ NBModelBinaryImage ]= {"ArgumentsPattern"->{_}};


CreateNBModelBinaryImage[] := NBModelBinaryImage[ NBModelBinaryImageNet ];


Sample[ NBModelBinaryImage[ net_ ] ] := Module[{nbm=NetTake[net,"reshapeoutput"]},
   Map[ rndBinary, nbm[ ConstantArray[0,{28,28}]]["Output"], {2} ] ];


Train[ NBModelBinaryImage[ net_ ], samples_ ] :=
   NBModelBinaryImage[ NetTrain[ net, Association[ "Input"->#, "Output"->#]&/@samples, LossFunction->"Loss" ] ];


LogDensity[ NBModelBinaryImage[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];


NBModelDiscreteImageNet = NetGraph[{
   "const"->ConstantArrayLayer[{28,28,10}],
   "softmax"->SoftmaxLayer[],
   "crossentropyloss"->CrossEntropyLossLayer["Probabilities"]
},{
   "const"->"softmax",
   NetPort["Input"]->NetPort[{"crossentropyloss","Target"}],
   NetPort[{"softmax","Output"}]->{NetPort[{"crossentropyloss","Input"}],NetPort["Output"]},
   NetPort[{"crossentropyloss","Loss"}]->NetPort["Loss"]
}];


Discretize[image_]:=Map[ReplacePart[ConstantArray[0,{10}],1+Round[#*9]->1]&,ImageData[image],{2}]


SyntaxInformation[ NBModelDiscreteImage ]= {"ArgumentsPattern"->{_}};


CreateNBModelDiscreteImage[] := NBModelDiscreteImage[ NBModelDiscreteImageNet ];


rndMult[probs_]:=RandomChoice[probs->Range[1,10]]


Sample[ NBModelDiscreteImage[ net_ ] ] := Module[{nbm=NetTake[net,{"const","softmax"}]},tmp=nbm;
   Map[ rndMult, nbm[], {2} ] ]/10.;


Train[ NBModelDiscreteImage[ net_ ], samples_ ] :=
   NBModelDiscreteImage[ NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss", MaxTrainingRounds->1000 ] ];


LogDensity[ NBModelDiscreteImage[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
