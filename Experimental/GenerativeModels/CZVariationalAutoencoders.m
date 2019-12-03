(* ::Package:: *)

(*
   Auto-Encoding Variational Bayes, 2014. Kingma & Welling.
   http: https://arxiv.org/abs/1312.6114

   This repo was helpful as VaE guide: https://jmetzen.github.io/2015-11-27/vae.html
*)


<<"Experimental/GenerativeModels/CZNBModels.m"


CZCreateEncoder[ inputUnits_, latentUnits_, h1_:500, h2_: 500 ] :=
   NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "mu"->latentUnits,
      "logvar"->latentUnits},{
      "h1"->"h2"->{"mu","logvar"},"mu"->NetPort["Mean"],"logvar"->NetPort["LogVar"]},
      "Input"->{inputUnits}];


CZCreateDecoder[ outputUnits_, outputModel_, h1_:500, h2_:500 ] :=
   NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "o"->outputUnits,
      "cond"->outputModel},{
      NetPort["Conditional"]->"h1",
      "h1"->"h2"->"o"->NetPort[{"cond","Conditional"}],
      NetPort[{"cond","Output"}]->NetPort["Output"],
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
   NetPort[{"encoder","Mean"}]->{NetPort[{"sampler","Mean"}],NetPort[{"kl_loss","Mean"}],NetPort["Mean"]},
   NetPort[{"encoder","LogVar"}]->{NetPort[{"sampler","LogVar"}],NetPort[{"kl_loss","LogVar"}],NetPort["LogVar"]},
   NetPort[{"sampler","Output"}]->NetPort[{"decoder","Conditional"}],
   NetPort[{"decoder","Output"}]->NetPort["Output"],
   {NetPort[{"decoder","Loss"}],NetPort[{"kl_loss","Loss"}]}->"total_loss"->NetPort["Loss"]
}];


CZSampleVaELatent[ latentUnits_ ] := RandomVariate@MultinormalDistribution[ ConstantArray[0,{latentUnits}], IdentityMatrix[latentUnits] ];


SyntaxInformation[ CZVaE ]= {"ArgumentsPattern"->{_,_,_}};


CZCreateVaEBinaryVector[ inputUnits_:784, latentUnits_:8 ] :=
   CZVaE[ CZBinaryVector[ inputUnits ], latentUnits, CZCreateVaENet[
      CZCreateEncoder[ inputUnits, latentUnits ],
      CZCreateDecoder[ inputUnits, CZGenerativeOutputLayer[ LogisticSigmoid, CrossEntropyLossLayer["Binary"]] ] ] ];


CZSample[ CZVaE[ CZBinaryVector[ inputUnits_ ], latentUnits_, vaeNet_ ] ] :=
   Module[{decoder=NetExtract[ vaeNet, "decoder" ], probMap },tmp=decoder;
   probMap = decoder[Association["Conditional"->CZSampleVaELatent[ latentUnits ],
      "Input"->ConstantArray[0,{inputUnits}] ] ]["Output"];
   CZSampleBinaryVector@probMap
];


CZCreateVaEBinaryImage[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] :=
   CZVaE[ CZBinaryImage[ imageDims ], latentUnits, CZCreateVaENet[ CZCreateEncoder[ imageDims[[1]]*imageDims[[2]], latentUnits ], CZCreateDecoder[ imageDims[[1]]*imageDims[[2]], CZGenerativeOutputLayer[ LogisticSigmoid, CrossEntropyLossLayer["Binary"]] ] ] ];


CZSample[ CZVaE[ CZBinaryImage[ imageDims_ ], latentUnits_, vaeNet_ ] ] :=
   Partition[ CZSample[ CZVaE[ CZBinaryVector[ imageDims[[1]]*imageDims[[2]] ], latentUnits, vaeNet ] ], imageDims[[2]]];


CZCreateEncoderDiscreteImage[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] :=
   NetChain[{FlattenLayer[],CZCreateEncoder[imageDims[[1]]*imageDims[[2]]*10, latentUnits]}]


CZCreateOutputDiscreteImage[ imageDims_:{28,28} ] := NetGraph[{
   ReshapeLayer[Append[imageDims, 10]],
   SoftmaxLayer[],
   CrossEntropyLossLayer["Probabilities"]},{
   NetPort["Conditional"]->1->2,
   NetPort["Input"]->NetPort[{3,"Target"}],
   2->{NetPort["Output"],NetPort[{3,"Input"}]}}];


CZCreateVaEDiscreteImage[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] :=
   CZVaE[ CZDiscreteImage[ imageDims ], latentUnits, CZCreateVaENet[ CZCreateEncoderDiscreteImage[ imageDims ], CZCreateDecoder[ imageDims[[1]]*imageDims[[2]]*10, CZCreateOutputDiscreteImage[ imageDims ] ] ] ];


CZSample[ CZVaE[ CZDiscreteImage[ imageDims_ ], latentUnits_, vaeNet_ ] ] :=
   CZSampleDiscreteImage@NetExtract[vaeNet,"decoder"][
      Association[ "Conditional"->CZSampleVaELatent[ latentUnits ], "Input"->ConstantArray[0,Append[imageDims,10]] ] ]["Output"]/10;


CZEncoder[ CZBinaryVector[ inputUnits_ ], input_ ] := input


CZEncoder[ CZBinaryImage[ imageDims_ ], input_ ] := Flatten[input]


CZEncoder[ CZDiscreteImage[ imageDims_ ], input_ ] := CZOneHot@input


CZTrain[ CZVaE[ inputType_, latentUnits_, vaeNet_ ], samples_ ] := Module[{trained, lossNet, f},
   f[assoc_] := MapThread[
      Association["Input"->CZEncoder[ inputType, #1],"RandomSample"->#2]&,
      {RandomSample[samples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];
   trained = NetTrain[ vaeNet, f, LossFunction->"Loss", "BatchSize"->128 ];
   CZVaE[ inputType, latentUnits, trained ]
];


CZLogDensity[ CZVaE[ inputType_, latentUnits_, vaeNet_ ], sample_ ] :=
   -vaeNet[ Association[ "Input"->CZEncoder[inputType, sample], "RandomSample"->ConstantArray[0,{latentUnits}] ] ][ "Loss" ]
