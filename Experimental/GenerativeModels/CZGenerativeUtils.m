(* ::Package:: *)

CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


CZOneHot[ image_ ] := Transpose[ Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {2} ], {2,3,1}];


(* Note we've fixed variance here, may be more sophisticated choice
*)
CZSampleRealVector[ betas_ ] := Map[ RandomVariate[NormalDistribution[#,1]]&, betas, {2}];


CZSampleBinary[ betas_ ] := Map[ RandomChoice[{1-#,#}->{0,1}]&, betas, {2}];


CZSampleDiscrete[ probs_ ] := Map[ RandomChoice[#->Range[1,10]]&, Transpose[probs,{3,1,2}], {2} ];


SyntaxInformation[ CZGenerativeModel ]= {"ArgumentsPattern"->{_,_,_,_}};


SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZRealVector ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZDiscreteImage ]= {"ArgumentsPattern"->{_}};


CZTrain[ CZGenerativeModel[ model_, inputType_, encoder_, net_ ], samples_ ] := Module[{trained, lossNet, f},
   rnd=RandomSample[ samples ];len=Round[Length[rnd]*.9];
   {trainingSet,validationSet}={rnd[[;;len]],rnd[[len+1;;]]};
   trainBatch[assoc_] :=
      Table[ Append[ Association[ If[model===CZPixelCNN, "Image", "Input" ] ->encoder[RandomChoice[trainingSet]]], If[ Head@model===CZVaE||Head@model===CZPixelVaE||Head@model===CZNadeVaE, "RandomSample"->CZSampleVaELatent[ model[[1]] ], {} ] ], {assoc["BatchSize"]} ];
   validBatch[assoc_] :=
      Table[ Append[ Association[ If[model===CZPixelCNN, "Image", "Input" ] ->encoder[RandomChoice[validationSet]]], If[ Head@model===CZVaE||Head@model===CZPixelVaE||Head@model===CZNadeVaE, "RandomSample"->CZSampleVaELatent[ model[[1]] ], {} ] ], {assoc["BatchSize"]} ];      
      tp1=trainBatch;
   trained = NetTrain[ net, trainBatch, ValidationSet->validBatch, LossFunction->"Loss", "BatchSize"->128,MaxTrainingRounds->10000,
      LearningRateMultipliers->Switch[
         model,
         CZPixelVaE[_], Flatten[Table[
            {{"decoder",5,"predict"<>ToString[k],"mask"}->0,{"decoder",5,"loss"<>ToString[k],"mask"}->0},{k,4}],1],
         CZPixelCNN, Flatten[Table[
         {{"condpixelcnn","predict"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,4}],1],
         CZNBModel,{},
         CZVaE[_],{},
         CZNade[], {},
         CZNormFlowModel,{},
         CZRealNVP,{},
         CZNadeVaE[_], {} ] ];
   CZGenerativeModel[ model,  inputType, encoder, trained ]
];


CZLogDensity[ CZGenerativeModel[ modelType_, modelInput_, encoder_, net_ ], sample_ ] :=
   -net[ Append[ Association[ If[modelType===CZPixelCNN, "Image", "Input" ] ->encoder@sample ], If[ Head@modelType===CZVaE||Head@modelType===CZPixelVaE||Head@modelType===CZNadeVaE, "RandomSample"->ConstantArray[0,{modelType[[1]]}], {} ] ] ]


(*
   Note CrossEntropyLossLayer computes everything in nats
*)
CZGenerativeOutputLayer[ outputLayerType_, lossType_, dims_ ] := NetGraph[{
   "out"->outputLayerType,
   "loss"->{lossType,ElementwiseLayer[#*Apply[Times,dims]&]}},{
   NetPort["Input"]->"out"->"loss"->NetPort["Loss"],
   NetPort["Target"]->NetPort[{"loss","Target"}]
}];


CZCrossEntropyLossLayer = NetGraph[{
   "cross"->CrossEntropyLossLayer["Probabilities"],
   "t1"->TransposeLayer[{3<->1,1<->2}],
   "t2"->TransposeLayer[{3<->1,1<->2}]},
   {NetPort["Input"]->"t1"->NetPort[{"cross","Input"}],
   NetPort["Target"]->"t2"->NetPort[{"cross","Target"}]}];


CZLossLayerWithTransfer[ CZBinary[ dims_ ] ] := CZGenerativeOutputLayer[ {PartLayer[1],LogisticSigmoid}, CrossEntropyLossLayer["Binary"], dims ]


CZLossLayerWithTransfer[ CZDiscrete[ dims_ ] ] := CZGenerativeOutputLayer[
   SoftmaxLayer[1], CZCrossEntropyLossLayer, dims ]


CZLossLayerWithTransfer[ CZRealGauss[ dims_ ] ] := CZGaussianLossLayer


(*
  borrowed from Nade module (needs refactoring), changed partlayer ordering
  and log normalisation term
*)
CZGaussianLossLayer = NetGraph[{
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
   "neglogpdf"->ElementwiseLayer[-#&],
   "loss"->SummationLayer[]},{
   NetPort["Target"]->"neg",
   {"mean","neg"}->"diff",
   "logdev"->"precision",
   {"diff","precision"}->"mult"->"sq"->"expterm",
   "precision"->"normterm",
   {"normterm","expterm"}->"logpdf"->"neglogpdf"->"loss"->NetPort["Loss"]
}];
