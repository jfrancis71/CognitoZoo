(* ::Package:: *)

(*
   Auto-Encoding Variational Bayes, 2014. Kingma & Welling.
   http: https://arxiv.org/abs/1312.6114

   This repo was helpful as VaE guide: https://jmetzen.github.io/2015-11-27/vae.html
*)


<<"Experimental/GenerativeModels/CZNBModels.m"


CZCreateEncoder[ dims_, latentUnits_, h1_:500, h2_: 500 ] :=
   NetGraph[{
      "h1"->{FlattenLayer[],h1,Ramp},
      "h2"->{h2,Ramp},
      "mu"->latentUnits,
      "logvar"->latentUnits},{
      "h1"->"h2"->{"mu","logvar"},"mu"->NetPort["Mean"],"logvar"->NetPort["LogVar"]}];


CZCreateNBModel[ dims_, outputType_ ] := Module[{probabilityParameters},
   probabilityParameters = Switch[
      Head[outputType],
      CZBinary,1,
      CZRealGauss,2,
      CZDiscrete,10,
      _,$Failed];
   NetGraph[{
      "array"->ConstantArrayLayer[Prepend[ dims, probabilityParameters ]],
      "loss"->CZLossLayerWithTransfer[ outputType ]},{
      "array"->NetPort[{"loss","Input"}],
      NetPort["Input"]->NetPort[{"loss","Target"}]
}]];



CZCreateDecoder[ dims_, outputType_, h1_:500, h2_:500 ] := Module[{probabilityParameters},
   probabilityParameters = Switch[
      Head[outputType],
      CZBinary,1,
      CZRealGauss,2,
      CZDiscrete,10,
      _,$Failed];
   NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "o"->dims[[1]]*dims[[2]]*probabilityParameters,
      "r"->ReshapeLayer[{probabilityParameters,dims[[1]],dims[[2]]}],
      "cond"->CZLossLayerWithTransfer[ outputType ]},{
      NetPort["Conditional"]->"h1",
      "h1"->"h2"->"o"->"r"->NetPort[{"cond","Input"}],
      NetPort[{"cond","Loss"}]->NetPort["Loss"]
}]];


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


CZSampleVaELatent[ latentUnits_ ] := RandomVariate@MultinormalDistribution[ ConstantArray[0,{latentUnits}], IdentityMatrix[latentUnits] ];


SyntaxInformation[ CZVaE ]= {"ArgumentsPattern"->{_}};


meanSquaredLossLayer = NetGraph[{
   "negtarget"->ElementwiseLayer[-#&],
   "diff"->ThreadingLayer[Plus],
   "sq"->ElementwiseLayer[#^2&]},{
   NetPort["Target"]->"negtarget",
   {NetPort["Input"],"negtarget"}->"diff"->"sq"->NetPort["Loss"]
}];


CZSample[ CZGenerativeModel[ CZVaE[ latentUnits_ ], CZRealVector[ inputUnits_ ], encoder_, vaeNet_ ] ] :=
   Module[{decoder=NetExtract[ vaeNet, "decoder" ], probMap },tmp=decoder;
   probMap = decoder[Association["Conditional"->CZSampleVaELatent[ latentUnits ],
      "Input"->ConstantArray[0,{inputUnits}] ] ]["Output"];
   CZSampleRealVector@probMap
];


CZCreateVaEBinary[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] :=
   CZGenerativeModel[ CZVaE[ latentUnits ] , CZBinary[ imageDims ], Identity,
      CZCreateVaENet[ CZCreateEncoder[ imageDims, latentUnits ], CZCreateDecoder[ imageDims, CZBinary[ imageDims ] ] ] ];


CZSample[ CZGenerativeModel[ CZVaE[ latentUnits_ ], CZBinary[ dims_ ], encoder_, vaeNet_ ] ] :=
   Module[{decoder=NetTake[NetFlatten[NetExtract[ vaeNet, "decoder" ] ],{NetPort["Conditional"],"cond/out/2"}], probMap },tmp=decoder;
   probMap = decoder[Association["Conditional"->CZSampleVaELatent[ latentUnits ] ] ];
   CZSampleBinary@probMap
];


CZCreateVaEDiscrete[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] :=
   CZGenerativeModel[ CZVaE[ latentUnits ],  CZDiscrete[ imageDims ], CZOneHot,
      CZCreateVaENet[ CZCreateEncoder[ imageDims, latentUnits ], CZCreateDecoder[ imageDims, CZDiscrete[ imageDims ] ] ] ];


CZSample[ CZGenerativeModel[ CZVaE[ latentUnits_ ], CZDiscrete[ imageDims_ ], encoder_, vaeNet_ ] ] :=
   CZSampleDiscrete@NetTake[NetFlatten[NetExtract[vaeNet,"decoder"]],"cond/out"][
      Association[ "Conditional"->CZSampleVaELatent[ latentUnits ]]]/10;


CZCreateVaERealGauss[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] :=
   CZGenerativeModel[ CZVaE[ latentUnits ],  CZRealGauss[ imageDims ], Identity,
      CZCreateVaENet[ CZCreateEncoder[ imageDims, latentUnits ], CZCreateDecoder[ imageDims, CZRealGauss[ imageDims ] ] ] ];


CZSample[ CZGenerativeModel[ CZVaE[ latentUnits_ ], CZRealGauss[ imageDims_ ], encoder_, vaeNet_ ] ] :=(
   mean=NetTake[NetFlatten[NetExtract[vaeNet,"decoder"]],"cond/mean"][
      Association[ "Conditional"->CZSampleVaELatent[ latentUnits ]]];
      logdev=NetTake[NetFlatten[NetExtract[vaeNet,"decoder"]],"cond/mean"][
      Association[ "Conditional"->CZSampleVaELatent[ latentUnits ]]];
        mean+Sqrt[Exp[logdev]]*Table[RandomVariate[NormalDistribution[0,1]],{imageDims[[1]]},{imageDims[[2]]}]
        )


CZGetLatent[ CZGenerativeModel[ CZVaE[ _ ], _, encoder_, vaeNet_ ], sample_ ] :=
 NetExtract[ vaeNet, "encoder"][ sample ]["Mean"];  


CZSampleFromLatent[ CZGenerativeModel[ CZVaE[ _ ], CZBinaryVector[ inputUnits_ ], encoder_, vaeNet_ ], latent_ ] :=
   Module[{decoder=NetExtract[ vaeNet, "decoder" ], probMap },tmp=decoder;
      probMap = decoder[Association["Conditional"->latent,
      "Input"->ConstantArray[0,{inputUnits}] ] ]["Output"];
   CZSampleBinaryVector@probMap
];
