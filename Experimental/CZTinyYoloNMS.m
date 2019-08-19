(* ::Package:: *)

(* Highly experimental.....Attempt to try building a tiny yolo based detector with conditional decoding *)


net=Import["~/Google Drive/Personal/Computer Science/CZModels/TinyNMSTrainingLocalisation/2019-08-09T09:48:16_0_016_12800_9.40e-3_1.03e-2.wlnet"];


SetDirectory["~/CognitoZoo"]


<<CZDetectObjects.m


nmsnet=NetTake[net,{"cond1","log"}];


trunknet=NetTake[net,{"base","base"}];


CZICMStep[ out_, sofar_ ] := ( 
r2=nmsnet[Association["1"->out,"cond"->sofar]];
m=(1-sofar)*r2;
If[Max[m]<.24,sofar,CZICMStep[out, ReplacePart[sofar,Position[m,Max[m]]->1]] ]) 


CZICM[ out_ ]:= (CZICMStep[ out, ConstantArray[0,{5,13,13}]])


CZOutput[threshold_:.24][image_]:=( 
netOutput=YoloNet[image];
slots=LogisticSigmoid[netOutput[[5;;105;;25]]]*SoftmaxLayer[][Transpose[Partition[netOutput,25][[All,6;;25]],{1,4,2,3}]];
nms=Position[CZICM[ trunknet[ image ] ],1];
Map[{CZGetBoundingBox[#,netOutput],Extract[CZPascalClasses,Ordering[Extract[slots,#],-1]],Max@Extract[slots,#]}&,nms]);


CZDetectObjectsNMS[ image_, opts:OptionsPattern[] ] :=
   CZObjectsDeconformer[ image, {416, 416}, "Fit" ]@CZOutput[ .24 ]@CZImageConformer[{416,416},"Fit"]@image;


CZHighlightObjectsNMS[img_,opts:OptionsPattern[]]:=HighlightImage[img,CZDisplayObjects@CZDetectObjectsNMS[img,opts]]


(*
   Returns a triple, first element is rectangle matchings, second element is unmatched elements from rects1
   and third element is unmatched elements from rects2
*)
CZMatchingRectangles[ rects1_, rects2_ ] := Module[ { matching = FindIndependentEdgeSet[ Graph[
   Flatten@Table[{1,i}->{2,j},{i,1,Length[rects1]},{j,1,Length[rects2]}],
   EdgeWeight->Flatten@Table[CZIntersectionOverUnion[rects1[[i]],rects2[[j]]],{i,1,Length[rects1]},{j,1,Length[rects2]}]]] },
   { Map[rects1[[#[[1,2]]]]->rects2[[#[[2,2]]]]&,matching], Delete[rects1,{#}&/@matching[[All,1,2]]], Delete[rects2,{#}&/@matching[[All,2,2]]] }];


CZErrorMatchingRectangles[ rects1_, rects2_ ] := Length[rects1]+Length[rects2]-2*Length[CZMatchingRectangles[rects1,rects2][[1]]] +
   Total@N@Map[1-CZIntersectionOverUnion[#[[1]],#[[2]]]&,CZMatchingRectangles[rects1,rects2][[1]]];


CZErrorMatchingObjects[ objs1_, objs2_ ] := CZErrorMatchingRectangles[ objs1[[All,1]], objs2[[All,1]] ]
