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


basenet = NetTake[NetExtract[TinyYoloNet,"trunkNet"],{1,31},"Input"->NetEncoder[{"Image",{416,416},ColorSpace->"RGB"}]];


base=Table[basenet[File["~/ImageDataSets/PascalVOC/VOC2012/ConformJPEGImages/"<>files[[randomOrdering[[k]]]]<>".jpg"]],{k,1,17125}];base//Dimensions


outnet = NetTake[NetExtract[TinyYoloNet,"trunkNet"],{1,32},"Input"->NetEncoder[{"Image",{416,416},ColorSpace->"RGB"}]];


dets=Table[Position[LogisticSigmoid[Partition[outnet[File["~/ImageDataSets/PascalVOC/VOC2012/ConformJPEGImages/"<>files[[randomOrdering[[k]]]]<>".jpg"]],25][[All,5]]],x_/;x>.24],{k,1,17125}];


comp[k_]:=NetChain[{ConstantTimesLayer["Scaling"->ReplacePart[ConstantArray[1,1924],{1024+400+k}->0]],1}]


nmsnet=NetGraph[Join[
Table[("cond"<>ToString[k])->comp[k],{k,1,100}],{"cat"->CatenateLayer[],"log"->LogisticSigmoid}],Append[Table["cond"<>ToString[k]->"cat",{k,1,100}],"cat"->"log"]];


extr[arr_,y_,x_]:=Table[If[dy>0&&dx>0&&dx<13&&dy<14,arr[[All,dy,dx]],ConstantArray[0,{100}]],{dy,y-1,y+1},{dx,x-1,x+1}]


strDataset=Table[
Module[{targs=CZEncodeTarget[groundTruth[[k]]]},th=targs;
Join[base[[k,All,det[[2]],det[[3]]]],Flatten@extr[targs,det[[2]],det[[3]]]]->targs[[All,det[[2]],det[[3]]]]],{k,1,17125},{det,dets[[k]]}];


dataset=Flatten[strDataset];


lrm=Table[{"cond"<>ToString[k],1,"Scaling"}->0,{k,1,100}];


{trainingSet,validationSet}={dataset[[;;95000]],dataset[[95001;;]]};


(*
   trained=NetTrain[nmsnet,trainingSet,ValidationSet\[Rule]validationSet,LearningRateMultipliers\[Rule]lrm]
*)
