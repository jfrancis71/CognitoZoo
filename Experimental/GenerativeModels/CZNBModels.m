(* ::Package:: *)

<<"Experimental/GenerativeModels/CZDiscreteImage.m"


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


CZCreateNBConditionalModel::method = "CZCreateNBConditionalModel method `1` should be one of BinaryVector, BinaryImage, DiscreteImage";


CZCreateNBConditionalModel[ type_ ] := NetGraph[{
   "out"->Switch[ type,
      "BinaryVector", LogisticSigmoid,
      "BinaryImage", LogisticSigmoid,
      "DiscreteImage", {TransposeLayer[{3<->1,1<->2}],SoftmaxLayer[]},
      _, (Message[ CZCreateNBConditionalModel::method, OptionValue[ Method ] ]; Abort[] ) ],
   "crossentropy"->Switch[ type,
      "BinaryVector", CrossEntropyLossLayer["Binary"],
      "BinaryImage", CrossEntropyLossLayer["Binary"],
      "DiscreteImage", CrossEntropyLossLayer["Index"],
      _, (Message[ CZCreateNBConditionalModel::method, OptionValue[ Method ] ]; Abort[] ) ]},{
   "out"->"crossentropy"->NetPort["Loss"],
   "out"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}],
   NetPort["Conditional"]->"out"
}];


CZCreateNBModel::method = "CZCreateNBModel method `1` should be one of BinaryVector, BinaryImage, DiscreteImage";


CZCreateNBModel[ type_ ] := NetGraph[{
   "array"->Switch[ type,
      "BinaryVector", ConstantArrayLayer[{784}],
      "BinaryImage", ConstantArrayLayer[{28,28}],
      "DiscreteImage", ConstantArrayLayer[{10,28,28}],
      _, (Message[ CZCreateNBModel::method, OptionValue[ Method ] ]; Abort[] ) ],
   "decoder"->CZCreateNBConditionalModel[ type ]},{
   "array"->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Input"}]
}];



SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{}};


CZCreateNBModelBinaryVector[] := CZNBModel[ CZBinaryVector, CZCreateNBModel[ "BinaryVector" ] ];


CZSample[ CZNBModel[ CZBinaryVector, net_ ] ] := Module[{out=net[ConstantArray[0,{784}]]["Output"]},
   rndBinary /@ out ];


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


CZCreateNBModelBinaryImage[] := CZNBModel[ CZBinaryImage, CZCreateNBModel[ "BinaryImage" ] ];


CZSample[ CZNBModel[ CZBinaryImage, net_ ] ] :=
   Map[ rndBinary, net[ ConstantArray[0,{28,28}]]["Output"], {2} ];


CZCreateNBModelDiscreteImage[] := CZNBModel[ CZDiscreteImage, CZCreateNBModel[ "DiscreteImage" ] ];


rndMult[probs_]:=RandomChoice[probs->Range[1,10]]


CZSample[ CZNBModel[ CZDiscreteImage, net_ ] ] :=
   Map[ rndMult, net[ConstantArray[1,{28,28}]]["Output"], {2} ]/10.;


SyntaxInformation[ CZNBModel ]= {"ArgumentsPattern"->{_,_}};


CZTrain[ CZNBModel[ modelType_, net_ ], samples_ ] :=
   CZNBModel[ modelType, NetTrain[ net, Association[ "Input"->#]&/@samples, LossFunction->"Loss", MaxTrainingRounds->1000 ] ];


CZLogDensity[ CZNBModel[ _, net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
