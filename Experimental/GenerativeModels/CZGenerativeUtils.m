(* ::Package:: *)

CZDiscretize[image_]:={Map[1+Round[#*9]&,ImageData[image, Interleaving->False],{2}]}


CZOneHot[ image_ ] := Transpose[ Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {Length[Dimensions[image]]} ],
   If[Length[Dimensions[image]]==2,{2,3,1},{2,3,4,1}]];


CZSampleDistribution[ CZBinary[ dims_ ], betas_ ] := Map[ RandomChoice[{1-#,#}->{0,1}]&, LogisticSigmoid@betas[[1]], {Length[dims]}];


CZSampleDistribution[ CZDiscrete[ dims_ ], probs_ ] := Map[ RandomChoice[#->Range[1,10]]&, SoftmaxLayer[]@
   Transpose[probs,If[Length[dims]==2,{3,1,2},{4,1,2,3}]], {Length[dims]} ];


CZSampleStandardNormalDistribution[ dims_ ] := Array[RandomVariate@NormalDistribution[0,1]&,dims];


CZSampleDistribution[ CZRealGauss[ dims_ ], params_ ] := params[[1]]+Sqrt[Exp[params[[2]]]]*CZSampleStandardNormalDistribution[ dims ];


CZDistributionParameters[ CZBinary[ _ ] ] := 1;
CZDistributionParameters[ CZDiscrete[ _ ] ] := 10;
CZDistributionParameters[ CZRealGauss[ _ ] ] := 2;


CZEncoder[ type_[_] ] := If[ type===CZDiscrete, CZOneHot, Identity ]


SyntaxInformation[ CZGenerativeModel ]= {"ArgumentsPattern"->{_,_,_}};


SyntaxInformation[ CZBinary ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZDiscrete ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZRealGauss ]= {"ArgumentsPattern"->{_}};


createAssociation[ inputType_, model_, example_ ] :=
   Association[
      If[ model===CZNBModel, Nothing, "Input"->example ],
      If[ CZLatentModelQ[ model ], "RandomSample"->CZSampleStandardNormalDistribution[ model[[1]] ], Nothing ],
      "Target"->If[Head[inputType] === CZDiscrete,CZOneHot@example, example] ]


CZTrain[ CZGenerativeModel[ model_, inputType_, net_ ], samples_, opts:OptionsPattern[] ] := Module[{trained, lossNet, f},
   rnd=RandomSample[ samples ];len=Round[Length[rnd]*.9];
   {trainingSet,validationSet}={rnd[[;;len]],rnd[[len+1;;]]};
   trainBatch[assoc_] :=
      Table[ createAssociation[ inputType, model, RandomChoice[trainingSet]], {assoc["BatchSize"]} ];
   validBatch[assoc_] :=
      Table[ createAssociation[ inputType, model, RandomChoice[validationSet]], {assoc["BatchSize"]} ];
      tp1=trainBatch;
   trained = NetTrain[ net, {trainBatch,"RoundLength"->Length[trainingSet]}, ValidationSet->{validBatch,"RoundLength"->Length[validationSet]}, LossFunction->"Loss", "BatchSize"->128,
      FilterRules[{opts}, Options[NetTrain]], LearningRateMultipliers->CZModelLRM[ model, inputType[[1]] ] ];
   CZGenerativeModel[ model,  inputType, trained ]
];


CZLogDensity[ CZGenerativeModel[ modelType_, modelInput_, net_ ], sample_ ] :=
   -net[ createAssociation[ modelInput, modelType, sample ] ]


(*
   Exactly like CrossEntropyLossLayer["Probabilities"] but calculated using first dimension
*)
CZCrossEntropyLossLayer = NetGraph[{
   "cross"->CrossEntropyLossLayer["Probabilities"],
   "t1"->TransposeLayer[{3<->1,1<->2}],
   "t2"->TransposeLayer[{3<->1,1<->2}]},
   {NetPort["Input"]->"t1"->NetPort[{"cross","Input"}],
   NetPort["Target"]->"t2"->NetPort[{"cross","Target"}]}];


CZLossLogits[ CZBinary[ dims_ ] ] := NetGraph[ {
   PartLayer[1],
   LogisticSigmoid,
   CrossEntropyLossLayer[ "Binary" ],
   ElementwiseLayer[#*Apply[Times,dims]&] },{
   1->2->NetPort[3,"Input"],
   NetPort[3,"Loss"]->4->NetPort["Loss"]}]


CZLossLogits[ CZDiscrete[ dims_ ] ] := NetGraph[ {
   SoftmaxLayer[1],
   CZCrossEntropyLossLayer,
   ElementwiseLayer[#*Apply[Times,dims]&] },{
   NetPort["Input"]->1->NetPort[{2,"Input"}],
   NetPort["Target"]->NetPort[{2,"Target"}],
   NetPort[{2,"Loss"}]->3->NetPort["Loss"]} ]


CZLossLogits[ CZRealGauss[ dims_ ] ] := CZGaussianLossLayer


(*
  borrowed from Nade module (needs refactoring), changed partlayer ordering
  and log normalisation term
*)
CZRawGaussianLossLayer = NetGraph[{
   "mean"->PartLayer[1],
   "logdev"->PartLayer[2],
   "neg"->ElementwiseLayer[-#&],
   "diff"->TotalLayer[],
   "precision"->ElementwiseLayer[1/Exp[#]&],
   "mult"->ThreadingLayer[Times],
   "sq"->ElementwiseLayer[#^2&],
   "expterm"->ElementwiseLayer[-.5*#&],
   "normterm"->ElementwiseLayer[Log[#]-Log[Sqrt[2 Pi]]&],
   "logpdf"->TotalLayer[],
   "neglogpdf"->ElementwiseLayer[-#&]},{
   NetPort["Target"]->"neg",
   {"mean","neg"}->"diff",
   "logdev"->"precision",
   {"diff","precision"}->"mult"->"sq"->"expterm",
   "precision"->"normterm",
   {"normterm","expterm"}->"logpdf"->"neglogpdf"->NetPort["Loss"]
}];
CZGaussianLossLayer = NetGraph[{
   "loss"->CZRawGaussianLossLayer,
   "total_loss"->SummationLayer[]},{
   NetPort["Input"]->NetPort[{"loss","Input"}],
   NetPort["Target"]->NetPort[{"loss","Target"}],
   NetPort[{"loss","Loss"}]->"total_loss"->NetPort["Loss"]
}];


MaskLayer[ mask_ ] := NetGraph[{
   "mask"->ConstantArrayLayer["Array"->mask],
   "thread"->ThreadingLayer[Times]},{
   {NetPort["Input"],"mask"}->"thread"}]
