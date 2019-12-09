(* ::Package:: *)

<<"Experimental/GenerativeModels/CZGenerativeUtils.m"


nadeNet=NetGraph[Flatten[{
   "proc1"->{ConstantArrayLayer[{1}],LogisticSigmoid},
   Table["proc"<>ToString[k]->{PartLayer[;;k-1],k,Tanh,k,Tanh,1,LogisticSigmoid},{k,2,8}],
   "cat"->CatenateLayer[],
   "loss"->CrossEntropyLossLayer["Binary"]
},1],
   {Table["proc"<>ToString[k]->"cat",{k,1,8}],
   NetPort["Input"]->NetPort[{"loss","Target"}],
   NetPort[{"cat","Output"}]->{NetPort["Output"],NetPort[{"loss","Input"}]}},
   "Input"->8
];


SyntaxInformation[ CZNade ]= {"ArgumentsPattern"->{}};


CZCreateNade[] := CZGenerativeModel[ CZNade[], CZBinaryVector, Identity, nadeNet ]


CZSample[ CZGenerativeModel[ CZNade[], bin_, id_, net_ ] ] := ( 
   inp=ConstantArray[0,{8}];
   out=net[ inp ];
   For[k=1,k<=8,k++,
   out=CZSampleBinaryVector@net[inp]["Output"];
   inp[[k]]=out[[k]]
   ];
   inp
)
