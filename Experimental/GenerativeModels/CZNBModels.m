(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


CZLatentModelQ[ CZNBModel ] = False;


CZCreateNBModelNet[ outputType_ ] := NetGraph[{
   "array"->ConstantArrayLayer[Prepend[ outputType[[1]], CZDistributionParameters[ outputType ] ]],
   "loss"->CZLossLogits[ outputType ]},{
   "array"->NetPort[{"loss","Input"}],
   NetPort["Input"]->NetPort[{"loss","Target"}]
}];


SyntaxInformation[ CZNBModel ]= {"ArgumentsPattern"->{}};


CZCreateNBModel[ type_:CZBinary[{28,28}] ] := CZGenerativeModel[ 
   CZNBModel, type, CZCreateNBModelNet[ type ]
];


CZSample[ CZGenerativeModel[ CZNBModel, outputType_, net_ ] ] :=
   CZSampleDistribution[ outputType, NetExtract[net, "array"][] ]/If[Head[outputType]===CZDiscrete,10,1];


CZModelLRM[ CZNBModel, _ ] := {}
