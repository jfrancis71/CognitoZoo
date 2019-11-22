(* ::Package:: *)

rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


NBConditionalModelBinaryVectorNet = NetGraph[{
   "log"->LogisticSigmoid,
   "crossentropy"->CrossEntropyLossLayer["Binary"]},{
   "log"->"crossentropy"->NetPort["Loss"],
   "log"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}],
   NetPort["Conditional"]->"log"
}];


NBModelBinaryVectorNet = NetGraph[{
   "array"->ConstantArrayLayer[{784}],
   "decoder"->NBConditionalModelBinaryVectorNet},{
   "array"->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Input"}]
}];


SyntaxInformation[ NBModelBinaryVector ]= {"ArgumentsPattern"->{_}};


CreateNBModelBinaryVector[] := NBModelBinaryVector[ NBModelBinaryVectorNet ];


Sample[ NBModelBinaryVector[ net_ ] ] := Module[{out=net[ConstantArray[0,{784}]]["Output"]},
   rndBinary /@ out ];


Train[ NBModelBinaryVector[ net_ ], samples_ ] :=
   NBModelBinaryVector[ NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss" ] ];


LogDensity[ NBModelBinaryVector[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];


NBConditionalModelBinaryImageNet = NetGraph[{
   "cond"->NBConditionalModelBinaryVectorNet,
   "reshapeConditional"->ReshapeLayer[{784}],
   "reshapeInput"->ReshapeLayer[{784}],
   "reshapeOutput"->ReshapeLayer[{28,28}]},{
   NetPort["Input"]->"reshapeInput"->NetPort[{"cond","Input"}],
   NetPort["Conditional"]->"reshapeConditional"->NetPort[{"cond","Conditional"}],
   NetPort[{"cond","Output"}]->"reshapeOutput"->NetPort["Output"]
}];


NBModelBinaryImageNet = NetGraph[{
   "cond"->NBConditionalModelBinaryImageNet,
   "array"->ConstantArrayLayer[{28,28}]},{
   "array"->NetPort[{"cond","Conditional"}],
   NetPort[{"cond","Loss"}]->NetPort["Loss"]
}];


SyntaxInformation[ NBModelBinaryImage ]= {"ArgumentsPattern"->{_}};


CreateNBModelBinaryImage[] := NBModelBinaryImage[ NBModelBinaryImageNet ];


Sample[ NBModelBinaryImage[ net_ ] ] :=
   Map[ rndBinary, net[ ConstantArray[0,{28,28}]]["Output"], {2} ];


(* I am not sure why we are needing to specify output here, it shouldn't form part of loss function
*)
Train[ NBModelBinaryImage[ net_ ], samples_ ] :=
   NBModelBinaryImage[ NetTrain[ net, Association[ "Input"->#,"Output"->#]&/@samples, LossFunction->"Loss" ] ];


LogDensity[ NBModelBinaryImage[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];


NBConditionalModelDiscreteImageNet = NetGraph[{
   "softmax"->SoftmaxLayer[],
   "crossentropyloss"->CrossEntropyLossLayer["Probabilities"]
},{
   NetPort["Conditional"]->"softmax",
   NetPort["Input"]->NetPort[{"crossentropyloss","Target"}],
   NetPort[{"softmax","Output"}]->{NetPort[{"crossentropyloss","Input"}],NetPort["Output"]},
   NetPort[{"crossentropyloss","Loss"}]->NetPort["Loss"]
}];


NBModelDiscreteImageNet = NetGraph[{
   "const"->ConstantArrayLayer[{28,28,10}],
   "cond"->NBConditionalModelDiscreteImageNet
},{
   "const"->NetPort[{"cond","Conditional"}],
   NetPort["Input"]->NetPort[{"cond","Input"}]
}];


Discretize[image_]:=Map[ReplacePart[ConstantArray[0,{10}],1+Round[#*9]->1]&,ImageData[image],{2}]


SyntaxInformation[ NBModelDiscreteImage ]= {"ArgumentsPattern"->{_}};


CreateNBModelDiscreteImage[] := NBModelDiscreteImage[ NBModelDiscreteImageNet ];


rndMult[probs_]:=RandomChoice[probs->Range[1,10]]


Sample[ NBModelDiscreteImage[ net_ ] ] :=
   Map[ rndMult, net[ConstantArray[0,{28,28,10}]]["Output"], {2} ]/10.;


Train[ NBModelDiscreteImage[ net_ ], samples_ ] :=
   NBModelDiscreteImage[ NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss", MaxTrainingRounds->1000 ] ];


LogDensity[ NBModelDiscreteImage[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
