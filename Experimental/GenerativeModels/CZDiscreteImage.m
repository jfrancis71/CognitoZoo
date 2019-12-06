(* ::Package:: *)

CZDiscretize[image_]:=Map[1+Round[#*9]&,ImageData[image],{2}]


CZOneHot[ image_ ] := Map[ ReplacePart[ ConstantArray[ 0, {10} ], #->1 ]&, image, {2} ];


CZSampleBinaryVector[ betas_ ] := RandomChoice[{1-#,#}->{0,1}]& /@ betas;


CZSampleBinaryImage[ betas_ ] := Map[ RandomChoice[{1-#,#}->{0,1}]&, betas, {2}];


CZSampleDiscreteImage[ probs_ ] := Map[ RandomChoice[#->Range[1,10]]&, probs, {2} ];


SyntaxInformation[ CZGenerativeModel ]= {"ArgumentsPattern"->{_,_,_,_}};


SyntaxInformation[ CZBinaryVector ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZBinaryImage ]= {"ArgumentsPattern"->{_}};


SyntaxInformation[ CZDiscreteImage ]= {"ArgumentsPattern"->{_}};


CZTrain[ CZGenerativeModel[ model_, inputType_, encoder_, net_ ], samples_ ] := Module[{trained, lossNet, f},
   f[assoc_] :=
      Table[ Append[ Association[ If[model===CZPixelCNN, "Image", "Input" ] ->encoder[RandomChoice[samples]]], If[ Head@model===CZVaE||Head@model===CZPixelVaE, "RandomSample"->CZSampleVaELatent[ model[[1]] ], {} ] ], {assoc["BatchSize"]} ];
   trained = NetTrain[ net, f, LossFunction->"Loss", "BatchSize"->128,MaxTrainingRounds->10000,
      tmp=LearningRateMultipliers->Switch[
         model,
         CZPixelVaE[_], Flatten[Table[
            {{"decoder",5,"predict"<>ToString[k],"mask"}->0,{"decoder",5,"loss"<>ToString[k],"mask"}->0},{k,4}],1],
         CZPixelCNN, Flatten[Table[
         {{"condpixelcnn","predict"<>ToString[k],"mask"}->0,{"condpixelcnn","loss"<>ToString[k],"mask"}->0},{k,4}],1],
         CZNBModel,{},
         CZVaE[_],{}] ];
   CZGenerativeModel[ model,  inputType, encoder, trained ]
];
