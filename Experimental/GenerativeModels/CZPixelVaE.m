(* ::Package:: *)

<<"Experimental/GenerativeModels/CZPixelCNN.m"


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


CZCreatePixelVaEBinaryImageDecoder[ imageDims_, h1_, h2_ ] := NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "o"->imageDims[[1]]*imageDims[[2]],
      "reshapecond"->ReshapeLayer[imageDims],
      "reshapeinput"->ReshapeLayer[imageDims],
      "cond"->CZConditionalPixelCNNBinaryImage[ imageDims ]},{
      NetPort["Conditional"]->"h1"->"h2"->"o"->"reshapecond"->NetPort[{"cond","Conditional"}],
      NetPort["Input"]->"reshapeinput"->NetPort[{"cond","Image"}]
}]


SyntaxInformation[ CZPixelVaEBinaryImage ]= {"ArgumentsPattern"->{_,_,_}};


CZCreatePixelVaEBinaryImage[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] := CZPixelVaEBinaryImage[
   imageDims,
   latentUnits,
   NetGraph[{
      "encoder"->CZCreateEncoder[ imageDims[[1]]*imageDims[[2]], latentUnits, h1, h2 ],
      "sampler"->CZVaESamplerNet,
      "decoder"->CZCreatePixelVaEBinaryImageDecoder[ imageDims, h1, h2 ],
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


CZTrain[ CZPixelVaEBinaryImage[ imageDims_, latentUnits_, net_ ], examples_ ] := (
   f[assoc_] := MapThread[
      Association["Input"->Flatten[#1],"RandomSample"->#2]&,
      {RandomSample[examples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];

   CZPixelVaEBinaryImage[ imageDims, latentUnits, NetTrain[ net, f,
      LearningRateMultipliers->
         Flatten[Table[
         {{"decoder","cond","conv"<>ToString[k],"mask"}->0,{"decoder","cond","loss"<>ToString[k],"mask"}->0},{k,1,4}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->{"kl_loss","recon_loss"},"BatchSize"->128 ]
] );


CZSample[ CZPixelVaEBinaryImage[ imageDims_, latentUnits_, net_ ] ] := Module[{s=ConstantArray[0,imageDims],decoder=NetExtract[ net, "decoder" ],cond=NetExtract[ net, {"decoder","cond"}]},tmp=decoder;
   znet = NetTake[ decoder, "reshapecond" ]; pixelOrdering=PixelCNNOrdering[ imageDims ];
   z = znet[ RandomVariate@MultinormalDistribution[ConstantArray[0,{latentUnits}],IdentityMatrix[latentUnits] ] ];
   For[k=1,k<=Length[pixelOrdering],k++,
      l = cond[Association["Image"->s,"Conditional"->z]]["Output"<>ToString[k]];
      s = Map[rndBinary,l,{2}]*pixelOrdering[[k]]+s;t=s;
   ];
   s]


CZCreatePixelVaEDiscreteImageDecoder[ imageDims_, h1_, h2_ ] := NetGraph[{
      "h1"->{h1,Ramp},
      "h2"->{h2,Ramp},
      "o"->imageDims[[1]]*imageDims[[2]],
      "reshapecond"->ReshapeLayer[Prepend[imageDims,1]],
      "reshapeinput"->ReshapeLayer[Prepend[imageDims,10]],
      "cond"->CZConditionalPixelCNNDiscreteImage[ imageDims ]},{
      NetPort["Conditional"]->"h1"->"h2"->"o"->"reshapecond"->NetPort[{"cond","Conditional"}],
      NetPort["Input"]->"reshapeinput"->NetPort[{"cond","Image"}]
}]


SyntaxInformation[ CZPixelVaEDiscreteImage ]= {"ArgumentsPattern"->{_,_,_}};


CZCreatePixelVaEDiscreteImageEncoder[ imageDims_, latentUnits_, h1_, h2_ ] := NetGraph[{
   "fl"->FlattenLayer[],
   "enc"->CZCreateEncoder[ imageDims[[1]]*imageDims[[2]]*10, latentUnits, h1, h2 ]},{
   "fl"->"enc"
}];


CZCreatePixelVaEDiscreteImage[ imageDims_:{28,28}, latentUnits_:8, h1_:500, h2_:500 ] := CZPixelVaEDiscreteImage[
   imageDims,
   latentUnits,
   NetGraph[{
      "encoder"->CZCreatePixelVaEDiscreteImageEncoder[ imageDims, latentUnits, h1, h2 ],
      "sampler"->CZVaESamplerNet,
      "decoder"->CZCreatePixelVaEDiscreteImageDecoder[ imageDims, h1, h2 ],
      "kl_loss"->CZKLLoss
      },{
      NetPort[{"encoder","Mean"}]->{NetPort[{"sampler","Mean"}],NetPort[{"kl_loss","Mean"}],NetPort["Mean"]},
      NetPort[{"encoder","LogVar"}]->{NetPort[{"sampler","LogVar"}],NetPort[{"kl_loss","LogVar"}],NetPort["LogVar"]},
      NetPort[{"sampler","Output"}]->NetPort[{"decoder","Conditional"}],
      NetPort["Input"]->NetPort[{"decoder"},"Input"],
      NetPort[{"decoder","Loss"}]->NetPort["recon_loss"],
      NetPort[{"kl_loss","Loss"}]->NetPort["kl_loss"]
   },"Input"->Prepend[imageDims,10]]
];


CZTrain[ CZPixelVaEDiscreteImage[ imageDims_, latentUnits_, net_ ], examples_ ] := (
   f[assoc_] := MapThread[
      Association["Input"->CZOneHot@#,"RandomSample"->#2]&,
      {RandomSample[examples,assoc["BatchSize"]],Partition[RandomVariate[NormalDistribution[0,1],latentUnits*assoc["BatchSize"]],latentUnits]}];

   CZPixelVaEDiscreteImage[ imageDims, latentUnits, NetTrain[ net, f,
      LearningRateMultipliers->
         Flatten[Table[
         {{"decoder","cond","conv"<>ToString[k],"mask"}->0,{"decoder","cond","loss"<>ToString[k],"mask"}->0},{k,1,4}],1]
      ,
      MaxTrainingRounds->10000,LossFunction->{"kl_loss","recon_loss"},"BatchSize"->128 ]
] );


CZSample[ CZPixelVaEDiscreteImage[ imageDims_, latentUnits_, net_ ] ] := Module[{s=ConstantArray[0,Prepend[imageDims,10]],decoder=NetExtract[ net, "decoder" ],cond=NetExtract[ net, {"decoder","cond"}]},
   znet = NetTake[ decoder, "reshapecond" ]; pixelOrdering=PixelCNNOrdering[ imageDims ];
   z = znet[ RandomVariate@MultinormalDistribution[ConstantArray[0,{latentUnits}],IdentityMatrix[latentUnits] ] ];c1=cond;
   For[k=1,k<=4,k++,
      l = cond[Association["Image"->s,"Conditional"->z]]["Output"<>ToString[k]];
      s = discreteSample[l]*Table[pixelOrdering[[k]],{10}]+s;t=s;
   ];
   Map[
Position[#,1][[1,1]]/10.&,
Transpose[s,{3,1,2}],{2}]
]
