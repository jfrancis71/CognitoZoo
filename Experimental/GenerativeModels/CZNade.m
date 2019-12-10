(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


condNadeNetNode[k_] := NetGraph[{PartLayer[;;k-1],CatenateLayer[],k,Tanh,k,Tanh,1,LogisticSigmoid},{
   NetPort["Input"]->1->2,
   NetPort["Conditional"]->2,
   2->3->4->5->6->7->8}]


createCondNadeNet[ inputUnits_ ] := NetGraph[Flatten[{
   "proc1"->{ConstantArrayLayer[{1}],LogisticSigmoid},
   Table["proc"<>ToString[k]->condNadeNetNode[k],{k,2,inputUnits}],
   "cat"->CatenateLayer[],
   "loss"->CrossEntropyLossLayer["Binary"]
},1],
   {Table["proc"<>ToString[k]->"cat",{k,1,inputUnits}],
   NetPort["Input"]->NetPort[{"loss","Target"}],
   NetPort[{"cat","Output"}]->{NetPort["Output"],NetPort[{"loss","Input"}]}},
   "Input"->inputUnits
];


createNadeNet[ inputUnits_ ] := NetGraph[{
   "constant"->ConstantArrayLayer[{1}],
   "condNadeNet"->createCondNadeNet[ inputUnits ]},{
   "constant"->NetPort[{"condNadeNet","Conditional"}]
   }];


SyntaxInformation[ CZNade ]= {"ArgumentsPattern"->{}};


CZCreateNade[ inputUnits_: 8] := CZGenerativeModel[ CZNade[], CZBinaryVector[ inputUnits ], Identity, createNadeNet[ inputUnits ] ]


CZSampleCondNade[ net_, cond_, inputUnits_ ] := (
   inp=ConstantArray[0,{inputUnits}];
   out=net[ Association[ "Input"->inp, "Conditional"->cond ] ];
   For[k=1,k<=inputUnits,k++,
   out=CZSampleBinaryVector@net[Association[ "Input"->inp, "Conditional"->cond ]]["Output"];
   inp[[k]]=out[[k]]
   ];
   inp
)


CZSample[ CZGenerativeModel[ CZNade[], CZBinaryVector[ inputUnits_ ], id_, net_ ] ] :=
   CZSampleCondNade[ NetExtract[ net, "condNadeNet" ], NetExtract[ net, "constant" ][], inputUnits ]


<<"Experimental/GenerativeModels/CZVariationalAutoencoders.m"


SyntaxInformation[ CZNadeVaE ]= {"ArgumentsPattern"->{_}};


CZCreateNadeVaE[ inputUnits_:8, latentUnits_:1 ] :=
   CZGenerativeModel[ CZNadeVaE[ latentUnits ], CZBinaryVector[ inputUnits ], Identity, CZCreateVaENet[
      CZCreateEncoder[ inputUnits, latentUnits ],
      CZCreateDecoder[ inputUnits, createCondNadeNet[ inputUnits ] ] ] ];


CZSample[ CZGenerativeModel[ CZNadeVaE[ latentUnits_ ], CZBinaryVector[ inputUnits_ ], id_, net_ ] ] :=
   CZSampleCondNade[ NetExtract[ net, {"decoder","cond"} ], NetTake[ NetExtract[ net, "decoder" ], {"h1","o"} ][ CZSampleVaELatent[ latentUnits ] ], inputUnits ]
