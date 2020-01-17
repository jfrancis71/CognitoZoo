(* ::Package:: *)

CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


CZOneHot[ image_ ] := Transpose[ Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {2} ], {2,3,1}];


CZSampleDistribution[ CZBinary[ dims_ ], betas_ ] := Map[ RandomChoice[{1-#,#}->{0,1}]&, LogisticSigmoid@betas[[1]], {2}];


CZSampleDistribution[ CZDiscrete[ dims_ ], probs_ ] := Map[ RandomChoice[#->Range[1,10]]&, SoftmaxLayer[][Transpose[probs,{3,1,2}]], {2} ];


CZSampleDistribution[ CZRealGauss[ dims_ ], params_ ] := params[[1]]+Sqrt[Exp[params[[2]]]]*Table[RandomVariate[NormalDistribution[0,1]],{Length[params[[1,1]]]},{Length[params[[1,2]]]}]


CZDistributionParameters[ CZBinary[ _ ] ] := 1;
CZDistributionParameters[ CZDiscrete[ _ ] ] := 10;
CZDistributionParameters[ CZRealGauss[ _ ] ] := 2;


CZEncoder[ type_[_] ] := If[ type===CZDiscrete, CZOneHot, Identity ]


SyntaxInformation[ CZGenerativeModel ]= {"ArgumentsPattern"->{_,_,_,_}};


SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZRealVector ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZDiscreteImage ]= {"ArgumentsPattern"->{_}};


Options[ CZTrain ] = {
   MaxTrainingRounds->10000 };
CZTrain[ CZGenerativeModel[ model_, inputType_, encoder_, net_ ], samples_, opts:OptionsPattern[] ] := Module[{trained, lossNet, f},
   rnd=RandomSample[ samples ];len=Round[Length[rnd]*.9];
   {trainingSet,validationSet}={rnd[[;;len]],rnd[[len+1;;]]};
   trainBatch[assoc_] :=
      Table[ Append[ Association[ "Input"->encoder[RandomChoice[trainingSet]]], If[ CZLatentModelQ[ model ], "RandomSample"->CZSampleVaELatent[ model[[1]] ], {} ] ], {assoc["BatchSize"]} ];
   validBatch[assoc_] :=
      Table[ Append[ Association[ "Input"->encoder[RandomChoice[validationSet]]], If[ CZLatentModelQ[ model ], "RandomSample"->CZSampleVaELatent[ model[[1]] ], {} ] ], {assoc["BatchSize"]} ];      
      tp1=trainBatch;
   trained = NetTrain[ net, trainBatch, ValidationSet->validBatch, LossFunction->"Loss", "BatchSize"->128,MaxTrainingRounds->OptionValue[ MaxTrainingRounds ],
      LearningRateMultipliers->CZModelLRM[ model ] ];
   CZGenerativeModel[ model,  inputType, encoder, trained ]
];


CZLogDensity[ CZGenerativeModel[ modelType_, modelInput_, encoder_, net_ ], sample_ ] :=
   -net[ Append[ Association[ "Input"->encoder@sample ], If[ Head@modelType===CZVaE||Head@modelType===CZPixelVaE||Head@modelType===CZNadeVaE, "RandomSample"->ConstantArray[0,{modelType[[1]]}], {} ] ] ]


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
