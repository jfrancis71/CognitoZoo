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


CZCreateVaE[ inputUnits_, latentUnits_, h1_:500, h2_:500 ] :=
   NetGraph[{
      "encoder"->CZCreateEncoder[ inputUnits, latentUnits, h1, h2 ],
      "sampler"->CZCreateSampler[],
      "decoder"->CZCreateDecoder[ inputUnits, h1, h2 ]},{
      NetPort[{"encoder","Mean"}]->NetPort[{"sampler","Mean"}],
      NetPort[{"encoder","LogVar"}]->NetPort[{"sampler","LogVar"}],
      NetPort[{"sampler","Output"}]->NetPort[{"decoder","Input"}],
      NetPort[{"encoder","Mean"}]->NetPort["Mean"],
      NetPort[{"encoder","LogVar"}]->NetPort["LogVar"]}];


CZCreateVaELoss[ inputUnits_, latentUnits_, h1_:500, h2_:500 ] :=
   NetGraph[{
      "VaE"->CZCreateVaE[ inputUnits, latentUnits, h1, h2 ],
      "recon_loss"->CrossEntropyLossLayer["Binary"],
      "exp"->ElementwiseLayer[Exp],
      "sq"->ElementwiseLayer[#^2&],
      "th"->ThreadingLayer[Plus],
      "neg"->ElementwiseLayer[-1-#&],
      "ag"->AggregationLayer[Total,1],
      "kl_loss"->ElementwiseLayer[0.5*#/784.&]},{
      NetPort[{"VaE","Output"}]->NetPort[{"recon_loss","Input"}],
      NetPort[{"VaE","LogVar"}]->"exp",
      NetPort[{"VaE","Mean"}]->"sq",
      NetPort[{"VaE","LogVar"}]->"neg",
      {"exp","sq","neg"}->"th"->"ag"->"kl_loss"->NetPort["kl_loss"],
      NetPort[{"recon_loss","Loss"}]->NetPort["recon_loss"]
}];


CZTrainVaE[ inputUnits_, latentUnits_, samples_, h1_:500, h2_:500 ] := Module[{trained, lossNet, f},
   f[assoc_] := MapThread[Association["Input"->#1,"Target"->#1,"RandomSample"->#2]&,{RandomSample[samples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];
   lossNet = CZCreateVaELoss[ inputUnits, latentUnits, h1, h2 ];
   trained = NetTrain[ lossNet, f, LossFunction->{"kl_loss", "recon_loss"} ];
   trained
];


CZGetEncoder[ vaeLossNet_ ] := NetExtract[ vaeLossNet, {"VaE", "encoder" } ]


CZGetDecoder[ vaeLossNet_ ] := NetExtract[ vaeLossNet, {"VaE", "decoder" } ]


(* trainedvaeloss = CZTrainVaE[ 784, 8, rawproc ] *)


(* dec=NetExtract[trainedvae, "decoder"] *)
