(* ::Package:: *)

(*
   Auto-Encoding Variational Bayes, 2014. Kingma & Welling.
   http: https://arxiv.org/abs/1312.6114

   This repo was helpful as VaE guide: https://jmetzen.github.io/2015-11-27/vae.html
*)


<<"Experimental/GenerativeModels/CZNBModels.m"


CZLatentModelQ[ CZVaE[ _ ] ] := True;


CZCreateEncoder[ latentUnits_, h1_:500, h2_: 500 ] :=
   NetGraph[{
      "h1"->{FlattenLayer[],h1,Ramp},
      "h2"->{h2,Ramp},
      "mu"->latentUnits,
      "logvar"->latentUnits},{
      "h1"->"h2"->{"mu","logvar"},"mu"->NetPort["Mean"],"logvar"->NetPort["LogVar"]}];


CZCreateDecoder[ outputType_, h1_:500, h2_:500 ] :=
   NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "o"->outputType[[1,1]]*outputType[[1,2]]*CZDistributionParameters[ outputType ],
      "r"->ReshapeLayer[{CZDistributionParameters[ outputType ],outputType[[1,1]],outputType[[1,2]]}],
      "cond"->CZLossLogits[ outputType ]},{
      NetPort["Conditional"]->"h1",
      "h1"->"h2"->"o"->"r"->NetPort[{"cond","Input"}],
      NetPort[{"cond","Loss"}]->NetPort["Loss"]
}];


CZVaESamplerNet = NetGraph[{
      "exp"->ElementwiseLayer[Exp[#/2]&],
      "times"->ThreadingLayer[Times],
      "sum"->TotalLayer[]},{
      NetPort["LogVar"]->"exp",
      {"exp",NetPort["RandomSample"]}->"times",
      {NetPort["Mean"],"times"}->"sum"}];


CZKLLoss = NetGraph[{
   "var"->ElementwiseLayer[Exp],
   "meansq"->ElementwiseLayer[#^2&],
   "th"->ThreadingLayer[Plus],
   "neg"->ElementwiseLayer[-1-#&],
   "ag"->AggregationLayer[Total,1],
   "kl_loss"->ElementwiseLayer[0.5*#&]},{
   NetPort["LogVar"]->{"var","neg"},
   NetPort["Mean"]->"meansq",
   {"var","meansq","neg"}->"th"->"ag"->"kl_loss"->NetPort["Loss"]
}];


CZCreateVaENet[ encoder_, decoder_ ] := NetGraph[{
   "encoder"->encoder,
   "sampler"->CZVaESamplerNet,
   "decoder"->decoder,
   "kl_loss"->CZKLLoss,
   "total_loss"->TotalLayer[]
   },{
   NetPort[{"encoder","Mean"}]->{NetPort[{"sampler","Mean"}],NetPort[{"kl_loss","Mean"}]},
   NetPort[{"encoder","LogVar"}]->{NetPort[{"sampler","LogVar"}],NetPort[{"kl_loss","LogVar"}]},
   NetPort[{"sampler","Output"}]->NetPort[{"decoder","Conditional"}],
   NetPort["Input"]->NetPort[{"decoder","Target"}],
   {NetPort[{"decoder","Loss"}],NetPort[{"kl_loss","Loss"}]}->"total_loss"->NetPort["Loss"]
}];


CZSampleVaELatent[ latentUnits_ ] := CZSampleStandardNormalDistribution[{1,latentUnits}][[1]];


SyntaxInformation[ CZVaE ]= {"ArgumentsPattern"->{_}};


CZSample[ CZGenerativeModel[ CZVaE[ latentUnits_ ], outputType_, vaeNet_ ] ] :=
   Module[{decoder=NetTake[NetExtract[ vaeNet, "decoder" ],{NetPort["Conditional"],"r"}], probMap },tmp=decoder;
   probMap = decoder[Association["Conditional"->CZSampleVaELatent[ latentUnits ] ] ];
   CZSampleDistribution[ outputType, probMap ]/If[Head[outputType]===CZDiscrete,10,1]
];


CZCreateVaE[ type_:CZBinary[{28,28}], latentUnits_:8, h1_:500, h2_:500 ] :=
   CZGenerativeModel[ CZVaE[ latentUnits ], type,
      CZCreateVaENet[ CZCreateEncoder[ latentUnits ], CZCreateDecoder[ type ] ] ];


CZGetLatent[ CZGenerativeModel[ CZVaE[ _ ], _, encoder_, vaeNet_ ], sample_ ] :=
 NetExtract[ vaeNet, "encoder"][ sample ]["Mean"];  


CZSampleFromLatent[ CZGenerativeModel[ CZVaE[ _ ], CZBinaryVector[ inputUnits_ ], encoder_, vaeNet_ ], latent_ ] :=
   Module[{decoder=NetExtract[ vaeNet, "decoder" ], probMap },tmp=decoder;
      probMap = decoder[Association["Conditional"->latent,
      "Input"->ConstantArray[0,{inputUnits}] ] ]["Output"];
   CZSampleBinaryVector@probMap
];


CZModelLRM[ CZVaE[ _ ] ] := {}
