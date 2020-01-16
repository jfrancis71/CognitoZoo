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


CZCreateNBModel[ dims_, outputType_ ] := Module[{probabilityParameters},
   probabilityParameters = Switch[
      Head[outputType],
      CZBinary,1,
      CZRealGauss,2,
      CZDiscrete,10,
      _,$Failed];
   NetGraph[{
      "array"->ConstantArrayLayer[Prepend[ dims, probabilityParameters ]],
      "loss"->CZLossLogits[ outputType ]},{
      "array"->NetPort[{"loss","Input"}],
      NetPort["Input"]->NetPort[{"loss","Target"}]
}]];


SyntaxInformation[ CZNBModel ]= {"ArgumentsPattern"->{}};


CZCreateNBModelBinary[ dims_:{28,28} ] := CZGenerativeModel[ 
   CZNBModel,
   CZBinary[ dims ], Identity,
   CZCreateNBModel[ dims, CZBinary[ dims ] ]
];


CZSample[ CZGenerativeModel[ CZNBModel, outputType_, _, net_ ] ] :=
   CZSampleDistribution[ outputType, NetExtract[net, "array"][] ]/If[Head[outputType]===CZDiscrete,10,1];


CZCreateNBModelDiscrete[ dims_:{28,28} ] := CZGenerativeModel[
   CZNBModel, CZDiscrete[ imageDims ], CZOneHot,
   CZCreateNBModel[ dims, CZDiscrete[ dims ] ]
];


CZCreateNBModelRealGauss[ dims_:{28,28} ] := CZGenerativeModel[
   CZNBModel, CZRealGauss[ dims ], Identity,
   CZCreateNBModel[ dims, CZRealGauss[ dims ] ]
];
