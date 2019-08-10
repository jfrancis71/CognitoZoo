(* ::Package:: *)

SetDirectory["~/CognitoZoo"];


<<CZTinyYoloV2Pascal.m


<<DataSetUtils/ImportPascalAnnotations.m


CZConformObjects[ {}, image_Image, netDims_List, "Fit" ] := {}


CZConformObjects[ objects_, image_Image, netDims_List, "Fit" ] :=
   Transpose@{CZConformRectangles[ objects[[All,1]], image, netDims, "Fit" ], objects[[All,2]] }


boundingboxes = Table[CZGetBoundingBox[ {l,y,x}, ConstantArray[0,{125,13,13}]],{l,1,5},{y,1,13},{x,1,13}];


returnmax[matrix_] := Position[matrix,Max[matrix]][[1]]


CZEncodeTarget[ objects_ ] :=
   ReplacePart[ConstantArray[0,{5,13,13}],
      Map[Function[{object},
         returnmax@Map[CZIntersectionOverUnion[object[[1]],#]&,boundingboxes,{3}]->1],objects]]


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


GTKernel = Flatten[Table[ReplacePart[ConstantArray[0,{5,3,3}],{{l,y,x}->1,{l,2,2}->0}],{l,1,5},{y,1,3},{x,1,3}],2];


basenet = NetTake[YoloNet,{1,31}];


nmsnet = NetGraph[{
   "base"->basenet,"cond1"->{ConvolutionLayer[45,{3,3},"PaddingSize"->1,"Weights"->GTKernel,"Biases"->None]},
   "bn"->ElementwiseLayer[#&],
   "cat"->CatenateLayer[],"c2"->{ConvolutionLayer[5,{1,1}]},"log"->LogisticSigmoid},{
   {"bn","cond1"}->"cat"->"c2"->"log",
   NetPort["Input"]->"base"->"bn",
   NetPort["cond"]->"cond1"}];


trained = NetTrain[
   nmsnet,dataset[[1;;16000]],ValidationSet->dataset[[16001;;]],
   LearningRateMultipliers->{{"cond1",1,"Weights"}->0,{"base",_}->0},
   MaxTrainingRounds->100,LearningRate->.001,
   TrainingProgressCheckpointing->{"Directory","~/Google Drive/Personal/Computer Science/CZModels/TinyNMSTrainingLocalisation/"},
   TrainingProgressReporting->{File["~/Google Drive/Personal/Computer Science/CZModels/TinyNMSTrainingLocalisation/results.csv"],"Interval"->Quantity[20,"Minutes"]}];
