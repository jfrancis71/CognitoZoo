(* ::Package:: *)

<<"Experimental/GenerativeModels/CZPixelCNN.m"


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


CZCreatePixelVaEBinaryImageDecoder[ inputUnits_, h1_, h2_ ] := NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "o"->inputUnits,
      "reshapecond"->ReshapeLayer[{28,28}],
      "reshapeinput"->ReshapeLayer[{28,28}],
      "cond"->CZConditionalPixelCNNBinaryImage},{
      NetPort["Conditional"]->"h1"->"h2"->"o"->"reshapecond"->NetPort[{"cond","Conditional"}],
      NetPort["Input"]->"reshapeinput"->NetPort[{"cond","Image"}]
}]


SyntaxInformation[ CZPixelVaEBinaryImage ]= {"ArgumentsPattern"->{_,_,_}};


CZCreatePixelVaEBinaryImage[ inputUnits_, latentUnits_, h1_:500, h2_:500 ] := CZPixelVaEBinaryImage[
   inputUnits,
   latentUnits,
   NetGraph[{
      "encoder"->CZCreateEncoder[ inputUnits, latentUnits, h1, h2 ],
      "sampler"->CZCreateSampler[],
      "decoder"->CZCreatePixelVaEBinaryImageDecoder[ inputUnits, h1, h2 ],
      "kl_loss"->CZKLLoss
      },{
      NetPort[{"encoder","Mean"}]->{NetPort[{"sampler","Mean"}],NetPort[{"kl_loss","Mean"}],NetPort["Mean"]},
      NetPort[{"encoder","LogVar"}]->{NetPort[{"sampler","LogVar"}],NetPort[{"kl_loss","LogVar"}],NetPort["LogVar"]},
      NetPort[{"sampler","Output"}]->NetPort[{"decoder","Conditional"}],
      NetPort["Input"]->NetPort[{"decoder"},"Input"],
      NetPort[{"decoder","Loss"}]->NetPort["recon_loss"],
      NetPort[{"kl_loss","Loss"}]->NetPort["kl_loss"]
   }]
];


CZTrain[ CZPixelVaEBinaryImage[ inputUnits_, latentUnits_, net_ ], examples_ ] := (
   f[assoc_] := MapThread[
      Association["Input"->Flatten[#1],"RandomSample"->#2]&,
      {RandomSample[examples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];

   CZPixelVaEBinaryImage[ inputUnits, latentUnits, NetTrain[ net, f,
      LearningRateMultipliers->
         Flatten[Table[
         {{"decoder","cond","conv"<>ToString[k],"mask"}->0,{"decoder","cond","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->{"kl_loss","recon_loss"},"BatchSize"->128 ]
] );


CZSample[ CZPixelVaEBinaryImage[ _, _, net_ ] ] := Module[{s=ConstantArray[0,{28,28}],decoder=NetExtract[ net, "decoder" ],cond=NetExtract[ net, {"decoder","cond"}]},tmp=decoder;
   znet = NetTake[ decoder, "reshapecond" ];
   z = znet[ RandomVariate@MultinormalDistribution[ConstantArray[0,{8}],IdentityMatrix[8] ] ];
   For[k=1,k<=Length[pixels],k++,
      l = cond[Association["Image"->s,"Conditional"->z]]["Output"<>ToString[k]][[1]];
      s = Map[rndBinary,l,{2}]*pixels[[k]][[1]]+s;t=s;
   ];
   s]


CZCreatePixelVaEDiscreteImageDecoder[ inputUnits_, h1_, h2_ ] := NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "o"->inputUnits,
      "reshapecond"->ReshapeLayer[{1,28,28}],
      "reshapeinput"->ReshapeLayer[{10,28,28}],
      "cond"->CZConditionalPixelCNNDiscreteImage},{
      NetPort["Conditional"]->"h1"->"h2"->"o"->"reshapecond"->NetPort[{"cond","Conditional"}],
      NetPort["Input"]->"reshapeinput"->NetPort[{"cond","Image"}]
}]


SyntaxInformation[ CZPixelVaEDiscreteImage ]= {"ArgumentsPattern"->{_,_,_}};


CZCreatePixelVaEDiscreteImageEncoder[ inputUnits_, latentUnits_, h1_, h2_ ] := NetGraph[{
   "fl"->FlattenLayer[],
   "enc"->CZCreateEncoder[ 7840, latentUnits, h1, h2 ]},{
   "fl"->"enc"
}];


CZCreatePixelVaEDiscreteImage[ inputUnits_, latentUnits_, h1_:500, h2_:500 ] := CZPixelVaEDiscreteImage[
   inputUnits,
   latentUnits,
   NetGraph[{
      "encoder"->CZCreatePixelVaEDiscreteImageEncoder[ inputUnits, latentUnits, h1, h2 ],
      "sampler"->CZCreateSampler[],
      "decoder"->CZCreatePixelVaEDiscreteImageDecoder[ inputUnits, h1, h2 ],
      "kl_loss"->CZKLLoss
      },{
      NetPort[{"encoder","Mean"}]->{NetPort[{"sampler","Mean"}],NetPort[{"kl_loss","Mean"}],NetPort["Mean"]},
      NetPort[{"encoder","LogVar"}]->{NetPort[{"sampler","LogVar"}],NetPort[{"kl_loss","LogVar"}],NetPort["LogVar"]},
      NetPort[{"sampler","Output"}]->NetPort[{"decoder","Conditional"}],
      NetPort["Input"]->NetPort[{"decoder"},"Input"],
      NetPort[{"decoder","Loss"}]->NetPort["recon_loss"],
      NetPort[{"kl_loss","Loss"}]->NetPort["kl_loss"]
   },"Input"->{10,28,28}]
];


CZTrain[ CZPixelVaEDiscreteImage[ inputUnits_, latentUnits_, net_ ], examples_ ] := (
   f[assoc_] := MapThread[
      Association["Input"->CZOneHot@#,"RandomSample"->#2]&,
      {RandomSample[examples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];

   CZPixelVaEDiscreteImage[ inputUnits, latentUnits, NetTrain[ net, f,
      LearningRateMultipliers->
         Flatten[Table[
         {{"decoder","cond","conv"<>ToString[k],"mask"}->0,{"decoder","cond","loss"<>ToString[k],"mask"}->0},{k,1,Length[pixels]}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->{"kl_loss","recon_loss"},"BatchSize"->128 ]
] );


CZSample[ CZPixelVaEDiscreteImage[ _, _, net_ ] ] := Module[{s=ConstantArray[0,{10,28,28}],decoder=NetExtract[ net, "decoder" ],cond=NetExtract[ net, {"decoder","cond"}]},
   znet = NetTake[ decoder, "reshapecond" ];
   z = znet[ RandomVariate@MultinormalDistribution[ConstantArray[0,{8}],IdentityMatrix[8] ] ];c1=cond;
   For[k=1,k<=4,k++,
      l = cond[Association["Image"->s,"Conditional"->z]]["Output"<>ToString[k]];
      s = discreteSample[l]*Table[pixels[[k]],{10}]+s;t=s;
   ];
   Map[
Position[#,1][[1,1]]/10.&,
Transpose[s,{3,1,2}],{2}]
]
