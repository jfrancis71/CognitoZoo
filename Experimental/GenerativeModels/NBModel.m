(* ::Package:: *)

rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


NBModelNet = NetGraph[{
   "array"->ConstantArrayLayer[{784}],
   "log"->LogisticSigmoid,
   "crossentropy"->CrossEntropyLossLayer["Binary"]},{
   "array"->"log"->"crossentropy"->NetPort["Loss"],
   "log"->NetPort["Output"],
   NetPort["Input"]->NetPort[{"crossentropy","Target"}]
}];


SyntaxInformation[ NBModel ]= {"ArgumentsPattern"->{_}};


CreateNBModel[] := NBModel[ NBModelNet ];


Sample[ NBModel[ net_ ] ] := Module[{out=NetTake[net,{"array","log"}]},t=out;
   rndBinary /@ out[] ];


Train[ NBModel[ net_ ], samples_ ] :=
   NBModel[ NetTrain[ net, Association[ "Input"->#, "Output"->#]&/@samples, LossFunction->"Loss" ] ];


LogDensity[ NBModel[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];


NBModelNetImage = NetGraph[{
   "nbmodel"->NBModelNet,
   "reshapeinput"->ReshapeLayer[{784}],
   "reshapeoutput"->ReshapeLayer[{28,28}]},{
   NetPort["Input"]->"reshapeinput"->NetPort[{"nbmodel","Input"}],
   NetPort[{"nbmodel","Output"}]->"reshapeoutput"->NetPort["Output"],
   NetPort[{"nbmodel","Loss"}]->NetPort["Loss"]
}];


SyntaxInformation[ NBModelImage ]= {"ArgumentsPattern"->{_}};


CreateNBModelImage[] := NBModelImage[ NBModelNetImage ];


Sample[ NBModelImage[ net_ ] ] := Module[{nbm=NetTake[net,"reshapeoutput"]},
   Map[ rndBinary, nbm[ ConstantArray[0,{28,28}]]["Output"], {2} ] ];


Train[ NBModelImage[ net_ ], samples_ ] :=
   NBModelImage[ NetTrain[ net, Association[ "Input"->#, "Output"->#]&/@samples, LossFunction->"Loss" ] ];


LogDensity[ NBModelImage[ net_ ], sample_ ] :=
   -net[ Association["Input" -> sample ] ]["Loss"];
