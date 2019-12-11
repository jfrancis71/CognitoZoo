(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


condNadeNetNode[ k_, outputType_ ] := NetGraph[{PartLayer[;;k-1],CatenateLayer[],k,Tanh,k,Tanh,1,outputType},{
   NetPort["Input"]->1->2,
   NetPort["Conditional"]->2,
   2->3->4->5->6->7->8}]


condNadeNetNodeRealVector[ k_ ] := NetGraph[{PartLayer[;;k-1],CatenateLayer[],k,Tanh,k,Tanh,2},{
   NetPort["Input"]->1->2,
   NetPort["Conditional"]->2,
   2->3->4->5->6->7}]


createCondNadeNet[ inputUnits_, outputType_, lossFunction_ ] := NetGraph[Flatten[{
   "proc1"->{ConstantArrayLayer[{1}],outputType},
   Table["proc"<>ToString[k]->condNadeNetNode[ k, outputType ],{k,2,inputUnits}],
   "cat"->CatenateLayer[],
   "loss"->lossFunction
},1],
   {Table["proc"<>ToString[k]->"cat",{k,1,inputUnits}],
   NetPort["Input"]->NetPort[{"loss","Target"}],
   NetPort[{"cat","Output"}]->{NetPort["Output"],NetPort[{"loss","Input"}]}},
   "Input"->inputUnits
];


CZGaussianLossLayer = NetGraph[{
   "mean"->PartLayer[{All,1}],
   "logdev"->PartLayer[{All,2}],
   "neg"->ElementwiseLayer[-#&],
   "diff"->TotalLayer[],
   "precision"->ElementwiseLayer[1/Exp[#]&],
   "mult"->ThreadingLayer[Times],
   "sq"->ElementwiseLayer[#^2&],
   "expterm"->ElementwiseLayer[-.5*#&],
   "normterm"->ElementwiseLayer[Log], (*I've ignored root pi term*)
   "logpdf"->TotalLayer[],
   "neglogpdf"->ElementwiseLayer[-#&],
   "loss"->SummationLayer[]},{
   NetPort["Target"]->"neg",
   {"mean","neg"}->"diff",
   "logdev"->"precision",
   {"diff","precision"}->"mult"->"sq"->"expterm",
   "precision"->"normterm",
   {"normterm","expterm"}->"logpdf"->"neglogpdf"->"loss"->NetPort["Loss"]
}];


createCondNadeRealVectorNet[ inputUnits_ ] := NetGraph[Flatten[{
   "proc1"->{ConstantArrayLayer[{2}]},
   Table["proc"<>ToString[k]->condNadeNetNodeRealVector[ k ],{k,2,inputUnits}],
   "cat"->CatenateLayer[],
   "reshape"->ReshapeLayer[{inputUnits,2}],
   "loss"->CZGaussianLossLayer
},1],
   {Table["proc"<>ToString[k]->"cat",{k,1,inputUnits}],
   NetPort["Input"]->NetPort[{"loss","Target"}],
   "cat"->"reshape",
   NetPort[{"reshape","Output"}]->{NetPort["Output"],NetPort[{"loss","Input"}]}},
   "Input"->inputUnits
];


createNadeNet[ inputUnits_, outputType_, lossFunction_ ] := NetGraph[{
   "constant"->ConstantArrayLayer[{1}],
   "condNadeNet"->createCondNadeNet[ inputUnits, outputType, lossFunction ]},{
   "constant"->NetPort[{"condNadeNet","Conditional"}]
   }];


createNadeRealVectorNet[ inputUnits_ ] := NetGraph[{
   "constant"->ConstantArrayLayer[{1}],
   "condNadeNet"->createCondNadeRealVectorNet[ inputUnits ]},{
   "constant"->NetPort[{"condNadeNet","Conditional"}]
   }];


SyntaxInformation[ CZNade ]= {"ArgumentsPattern"->{}};


CZCreateNadeBinaryVector[ inputUnits_: 8] := CZGenerativeModel[ CZNade[], CZBinaryVector[ inputUnits ], Identity, createNadeNet[ inputUnits, LogisticSigmoid, CrossEntropyLossLayer[ "Binary" ] ] ]


CZCreateNadeRealVector[ inputUnits_: 8] := CZGenerativeModel[ CZNade[], CZRealVector[ inputUnits ], Identity, createNadeRealVectorNet[ inputUnits ] ]


CZSampleCondNadeBinaryVector[ net_, cond_, inputUnits_ ] := (
   inp=ConstantArray[0,{inputUnits}];
   out=net[ Association[ "Input"->inp, "Conditional"->cond ] ];
   For[k=1,k<=inputUnits,k++,
   out=CZSampleBinaryVector@net[Association[ "Input"->inp, "Conditional"->cond ]]["Output"];
   inp[[k]]=out[[k]]
   ];
   inp
)


CZSampleCondNadeRealVector[ net_, cond_, inputUnits_ ] := (
   inp=ConstantArray[0,{inputUnits}];
   out=net[ Association[ "Input"->inp, "Conditional"->cond ] ];
   For[k=1,k<=inputUnits,k++,
   outm=net[Association[ "Input"->inp, "Conditional"->cond ]]["Output"][[k]];
   s=RandomVariate[NormalDistribution[outm[[1]],Exp@outm[[2]]]];
   inp[[k]]=s
   ];
   inp
)


CZSample[ CZGenerativeModel[ CZNade[], CZBinaryVector[ inputUnits_ ], id_, net_ ] ] :=
   CZSampleCondNadeBinaryVector[ NetExtract[ net, "condNadeNet" ], NetExtract[ net, "constant" ][], inputUnits ]


CZSample[ CZGenerativeModel[ CZNade[], CZRealVector[ inputUnits_ ], id_, net_ ] ] :=
   CZSampleCondNadeRealVector[ NetExtract[ net, "condNadeNet" ], NetExtract[ net, "constant" ][], inputUnits ]


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


SyntaxInformation[ CZNadeVaE ]= {"ArgumentsPattern"->{_}};


CZCreateNadeVaE[ inputUnits_:8, latentUnits_:1 ] :=
   CZGenerativeModel[ CZNadeVaE[ latentUnits ], CZBinaryVector[ inputUnits ], Identity, CZCreateVaENet[
      CZCreateEncoder[ inputUnits, latentUnits ],
      CZCreateDecoder[ inputUnits, createCondNadeNet[ inputUnits ] ] ] ];


CZSample[ CZGenerativeModel[ CZNadeVaE[ latentUnits_ ], CZBinaryVector[ inputUnits_ ], id_, net_ ] ] :=
   CZSampleCondNade[ NetExtract[ net, {"decoder","cond"} ], NetTake[ NetExtract[ net, "decoder" ], {"h1","o"} ][ CZSampleVaELatent[ latentUnits ] ], inputUnits ]
