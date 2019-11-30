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


SyntaxInformation[ CZVaEBinaryVector ]= {"ArgumentsPattern"->{_,_}};


CZCreateVaENet[ encoder_, decoder_ ] :=
   NetGraph[{
      "encoder"->encoder,
      "sampler"->CZVaESamplerNet,
      "decoder"->decoder,
      "kl_loss"->CZKLLoss
      },{
      NetPort[{"encoder","Mean"}]->{NetPort[{"sampler","Mean"}],NetPort[{"kl_loss","Mean"}],NetPort["Mean"]},
      NetPort[{"encoder","LogVar"}]->{NetPort[{"sampler","LogVar"}],NetPort[{"kl_loss","LogVar"}],NetPort["LogVar"]},
      NetPort[{"sampler","Output"}]->NetPort[{"decoder","Conditional"}],
      NetPort[{"decoder","Output"}]->NetPort["Output"],
      NetPort[{"decoder","Loss"}]->NetPort["recon_loss"],
      NetPort[{"kl_loss","Loss"}]->NetPort["kl_loss"]
}];


CZCreateVaEBinaryVector[ latentUnits_:8 ] :=
   CZVaEBinaryVector[ latentUnits, CZCreateVaENet[
      CZCreateEncoder[ 784, latentUnits ],
      CZCreateDecoder[ 784, CZGenerativeOutputLayer[ LogisticSigmoid, CrossEntropyLossLayer["Binary"]] ] ] ];


CZTrain[ CZVaEBinaryVector[ latentUnits_, vaeNet_ ], samples_ ] := Module[{trained, lossNet, f},
   f[assoc_] := MapThread[
      Association["Input"->#1,"RandomSample"->#2]&,
      {RandomSample[samples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];
   trained = NetTrain[ vaeNet, f, LossFunction->{"kl_loss", "recon_loss"}, "BatchSize"->128 ];
   CZVaEBinaryVector[ latentUnits, trained ]
];


CZLogDensity[ CZVaEBinaryVector[ latentUnits_, vaeNet_ ], sample_ ] :=
   Module[{proc=vaeNet[ Association["Input"->sample, "RandomSample"->ConstantArray[0, latentUnits ] ] ]},
      -(proc["kl_loss"]+proc["recon_loss"])]


rndBinary[beta_]:=RandomChoice[{1-beta,beta}->{0,1}];


CZSample[ CZVaEBinaryVector[ latentUnits_, vaeNet_ ] ] :=
   Module[{decoder=NetExtract[ vaeNet, "decoder" ], probMap },tmp=decoder;
   probMap = decoder[Association["Conditional"->
      RandomVariate@MultinormalDistribution[ ConstantArray[0, latentUnits ], IdentityMatrix[ latentUnits ] ],
      "Input"->ConstantArray[0,{784}] ] ]["Output"];
   rndBinary /@ probMap
];


SyntaxInformation[ CZVaEBinaryImage ]= {"ArgumentsPattern"->{_,_}};


CZCreateVaEBinaryImage[ latentUnits_:8, h1_:500, h2_:500 ] :=
   CZVaEBinaryImage[ latentUnits, CZCreateVaENet[ CZCreateEncoder[ 784, latentUnits ], CZCreateDecoder[ 784, CZGenerativeOutputLayer[ LogisticSigmoid, CrossEntropyLossLayer["Binary"]] ] ] ];


CZTrain[ CZVaEBinaryImage[ latentUnits_, vaeNet_ ], samples_ ] :=
   ReplacePart[ CZTrain[ CZVaEBinaryVector[ latentUnits, vaeNet ], Flatten/@samples ], 0->CZVaEBinaryImage ];


CZLogDensity[ CZVaEBinaryImage[ latentUnits_, vaeNet_ ], sample_ ] :=
   CZLogDensity[ CZVaEBinaryVector[ latentUnits, vaeNet ], Flatten@sample ] 


CZSample[ CZVaEBinaryImage[ latentUnits_, vaeNet_ ] ] :=
   Partition[ CZSample[ CZVaEBinaryVector[ latentUnits, vaeNet ] ], 28];


CZCreateEncoderDiscreteImage[ latentUnits_:8, h1_:500, h2_:500 ] :=
   NetChain[{FlattenLayer[],CZCreateEncoder[784*10, latentUnits]}]


CZCreateOutputDiscreteImage = NetGraph[{
   ReshapeLayer[{10,28,28}],
   SoftmaxLayer[1],
   CrossEntropyLossLayer["Probabilities"]},{
   NetPort["Conditional"]->1->2,
   NetPort["Input"]->NetPort[{3,"Target"}],
   2->{NetPort["Output"],NetPort[{3,"Input"}]}}];


SyntaxInformation[ CZVaEDiscreteImage ]= {"ArgumentsPattern"->{_,_}};


CZCreateVaEDiscreteImage[ latentUnits_:8, h1_:500, h2_:500 ] :=
   CZVaEDiscreteImage[ latentUnits, CZCreateVaENet[ CZCreateEncoderDiscreteImage[], CZCreateDecoder[ 784*10, CZCreateOutputDiscreteImage ] ] ];


rndMult[probs_]:=RandomChoice[probs->Range[1,10]]


CZTrain[ CZVaEDiscreteImage[ latentUnits_, vaeNet_ ], samples_ ] := Module[{trained, lossNet, f},
   f[assoc_] := MapThread[
      Association["Input"->(CZOneHot@#1),"RandomSample"->#2]&,
      {RandomSample[samples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];tmp=f;
   trained = NetTrain[ vaeNet, f, LossFunction->{"kl_loss", "recon_loss"}, "BatchSize"->128, MaxTrainingRounds->10000 ];
   CZVaEDiscreteImage[ latentUnits, trained ]
];


CZSample[ CZVaEDiscreteImage[ latentUnits_, vaeNet_ ] ] := (
   decoder=NetExtract[vaeNet,"decoder"];
   l1=discreteSample@decoder[Association[ "Conditional"->RandomVariate[MultinormalDistribution[ConstantArray[0,{latentUnits}],IdentityMatrix[latentUnits]]], "Input"->ConstantArray[0,{10,28,28}]]]["Output"];
      Map[
Position[#,1][[1,1]]/10.&,
Transpose[l1,{3,1,2}],{2}]
)


CZLogDensity[ CZVaEDiscreteImage[ latentUnits_, vaeNet_ ], sample_ ] := (
   r=vaeNet[ Association[ "Input"->CZOneHot@sample, "RandomSample"->ConstantArray[0,{latentUnits}] ] ];
   r["kl_loss"]+r["recon_loss"]
);
