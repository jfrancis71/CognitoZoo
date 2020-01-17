(* ::Package:: *)

(*
   Generative models all have an Input port, an Output port and a Loss port.
   Conditional Generative models are considered to be generative models, but have
   an additional Conditional port.
   The input for discrete images in the net is eg 28x28x10
   Note this isn't always convenient for within the net, so they may reformat (to 10x28x28).
   Models are in format: CZGenerativeModel[ modelType, inputType, encoder, net ]
*)


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


CZModelLRM[ CZNBModel ] := {}
