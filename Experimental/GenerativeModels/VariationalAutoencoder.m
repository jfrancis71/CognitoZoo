(* ::Package:: *)

(*
   Auto-Encoding Variational Bayes, 2014. Kingma & Welling.
   http: https://arxiv.org/abs/1312.6114

   This repo was helpful as VaE guide: https://jmetzen.github.io/2015-11-27/vae.html
*)


CZCreateEncoder[ inputUnits_, latentUnits_, h1_:500, h2_: 500 ] :=
   NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "mu"->latentUnits,
      "logvar"->latentUnits},{
      "h1"->"h2"->{"mu","logvar"},"mu"->NetPort["Mean"],"logvar"->NetPort["LogVar"]},
      "Input"->{inputUnits}];


CZCreateDecoder[ outputUnits_, h1_:500, h2_:500 ] :=
   NetChain[{
      {h1,Ramp},
      {h2,Ramp},
      outputUnits,
      LogisticSigmoid}];


CZCreateSampler[] :=
   NetGraph[{
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


SyntaxInformation[ VariationalAutoencoder ]= {"ArgumentsPattern"->{_,_,_}};


CZCreateVaE[ inputUnits_, latentUnits_, h1_:500, h2_:500 ] := VariationalAutoencoder[
   inputUnits,
   latentUnits,
   NetGraph[{
      "encoder"->CZCreateEncoder[ inputUnits, latentUnits, h1, h2 ],
      "sampler"->CZCreateSampler[],
      "decoder"->CZCreateDecoder[ inputUnits, h1, h2 ],
      "kl_loss"->CZKLLoss,
      "mean_recon_loss"->CrossEntropyLossLayer["Binary"],
      "total_recon_loss"->ElementwiseLayer[#*inputUnits&]
      },{
      NetPort[{"encoder","Mean"}]->{NetPort[{"sampler","Mean"}],NetPort[{"kl_loss","Mean"}],NetPort["Mean"]},
      NetPort[{"encoder","LogVar"}]->{NetPort[{"sampler","LogVar"}],NetPort[{"kl_loss","LogVar"}],NetPort["LogVar"]},
      NetPort[{"sampler","Output"}]->NetPort[{"decoder","Input"}],
      NetPort[{"decoder","Output"}]->{NetPort["Output"],NetPort[{"mean_recon_loss","Input"}]},
      NetPort["Input"]->NetPort[{"mean_recon_loss","Target"}],
      NetPort[{"mean_recon_loss","Loss"}]->"total_recon_loss"->NetPort["recon_loss"],
      NetPort[{"kl_loss","Loss"}]->NetPort["kl_loss"]
   }]
];


CZTrainVaE[ VariationalAutoencoder[ inputUnits_, latentUnits_, vae_ ], samples_ ] := Module[{trained, lossNet, f},
   f[assoc_] := MapThread[
      Association["Input"->#1,"RandomSample"->#2]&,
      {RandomSample[samples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];
   trained = NetTrain[ vae, f, LossFunction->{"kl_loss", "recon_loss"}, "BatchSize"->128 ];
   VariationalAutoencoder[ inputUnits, latentUnits, trained ]
];


CZLogDensity[ VariationalAutoencoder[ inputUnits_, latentUnits_, vae_ ], sample_ ] :=
   Module[{proc=vae[ Association["Input"->sample, "RandomSample"->ConstantArray[0, latentUnits ] ] ]},
      -(proc["kl_loss"]+proc["recon_loss"])]


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


CZSample[ VariationalAutoencoder[ inputUnits_, latentUnits_, vae_ ] ] :=
   Module[{decoder=NetExtract[ vae, "decoder" ], probMap },
   probMap = decoder[
      RandomVariate@MultinormalDistribution[ ConstantArray[0, latentUnits ], IdentityMatrix[ latentUnits ] ] ];
   rndBinary /@ probMap
];
