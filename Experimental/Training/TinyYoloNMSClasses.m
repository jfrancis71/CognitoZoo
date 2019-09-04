(* ::Package:: *)

SetDirectory["~/CognitoZoo"];


<<DataSetUtils/ImportPascalAnnotations.m


<<CZDetectObjects.m


CZConformObjects[ {}, image_Image, netDims_List, "Fit" ] := {}


CZConformObjects[ objects_, image_Image, netDims_List, "Fit" ] :=
   Transpose@{CZConformRectangles[ objects[[All,1]], image, netDims, "Fit" ], objects[[All,2]] }


biases={{1.08,1.19},{3.42,4.41},{6.63,11.38},{9.42,5.11},{16.62,10.52}};


boundingboxes = Table[Module[{cx=(x+.5)/13,cy=(y+.5)/13,width=biases[[l,1]]/13,height=biases[[l,2]]/13},Rectangle[416*{cx-width/2,1-cy-height/2},416*{cx+width/2,1-cy+height/2}]],{l,1,5},{y,0,12},{x,0,12}];


returnmax[matrix_] := Position[matrix,Max[matrix]][[1]]


CZEncodeTarget[ objects_ ] := Flatten[
   ReplacePart[ConstantArray[0,{5,20,13,13}],
      Map[Function[{object},tp=
         Module[{box = returnmax@Map[CZIntersectionOverUnion[object[[1]],#]&,boundingboxes,{3}] }, {box[[1]],Position[CZPascalClasses,object[[2]]][[1,1]],box[[2]],box[[3]]}]->1],objects]],1]


files = FileBaseName/@FileNames["~/ImageDataSets/PascalVOC/VOC2012/JPEGImages/*.jpg"];


randomOrdering = Import["~/ImageDataSets/PascalVOC/VOC2012/RandomOrdering.mx"];


groundTruth = Table[CZConformObjects[
   CZImportPascalAnnotations["~/ImageDataSets/PascalVOC/VOC2012/Annotations/"<>files[[randomOrdering[[k]]]]<>".xml"],
   Import["~/ImageDataSets/PascalVOC/VOC2012/JPEGImages/"<>files[[randomOrdering[[k]]]]<>".jpg"] , {416,416}, "Fit"],
   {k,1,17125}];


dataset = Table[
   lay=N@CZEncodeTarget[ groundTruth[[k]] ];
   Association[
      "Input"->File["~/ImageDataSets/PascalVOC/VOC2012/ConformJPEGImages/"<>files[[randomOrdering[[k]]]]<>".jpg"],
      "Output"->lay,
      "cond"->lay]
   ,{k,1,17125}];


GTKernel = Table[Flatten[Table[ReplacePart[ConstantArray[0,{100,3,3}],{{l,y,x}->1,{c,2,2}->0}],{l,1,100},{y,1,3},{x,1,3}],2],{c,1,100}];


basenet = NetTake[NetExtract[TinyYoloNet,"trunkNet"],{1,31},"Input"->NetEncoder[{"Image",{416,416},ColorSpace->"RGB"}]];


condGraph[kernel_] := NetGraph[{
   ConvolutionLayer[900,{3,3},"PaddingSize"->1,"Weights"->kernel,"Biases"->None],
   CatenateLayer[],
   ConvolutionLayer[1,{1,1}],
   LogisticSigmoid},{
   NetPort["Input"]->2,NetPort["cond"]->1->2,2->3->4}]


nmsnet = NetGraph[{
   "base"->basenet,
   Sequence@@Table["cond"<>ToString[k]->condGraph[GTKernel[[k]]],{k,1,100}],
   "cat"->CatenateLayer[]},{
   "base"->Table["cond"<>ToString[k],{k,1,100}]->"cat"
}];


trained = NetTrain[
   nmsnet,dataset[[1;;16000]],ValidationSet->dataset[[16001;;]],
   LearningRateMultipliers->Append[
      Table[{"cond"<>ToString[k],1,"Weights"}->0,{k,1,100}],{"base",_}->0],
   MaxTrainingRounds->100,LearningRate->.001,
   TrainingProgressCheckpointing->{"Directory","~/Google Drive/Personal/Computer Science/CZModels/TinyNMSTrainingClasses1/"},
   TrainingProgressReporting->{File["~/Google Drive/Personal/Computer Science/CZModels/TinyNMSTrainingClasses1/results.csv"],"Interval"->Quantity[20,"Minutes"]}];

(*
*)
